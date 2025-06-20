#
# This code simulates the effectiveness of an ideal 2x2 XTC filter configuration.
# It loads a provided SOFA file and extracts two HRIR sets for ±30° azimuth.
# It then generates 2x2 MIMO XTC filters and applies them two two ideal simulated
# loudspeakers in space, with geometry matching that of the HRIRs.
# It also computes the energy distribution of the crosstalk cancellation across a 
# 4x4 meter grid, which provides information about how the HRIRs translate to filter
# performance in space. At the moment, it doesn't really show substantial XTC at the
# locations we expect, although I think this is due to shortcomings in the simulation
# such as the lack of a head model, no 1/r2 attenuation, and the realities of an actual
# HRIR from the dataset.
#

import os
import time
import numpy as np
import random
#from pythonosc.udp_client import SimpleUDPClient
import threading
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QComboBox, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer
from pythonosc import dispatcher, osc_server
from scipy.fft import fft
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from math import sin, cos, radians
import matplotlib.pyplot as plt
from scipy.signal import oaconvolve
import xtc
import atexit
atexit.register(xtc.generate_cache_stats_plot)
xtc.init_filter_cache()
import level_meter
from audio_engine import AudioEngine

# --- Patch: Import wavwrite for saving WAV files ---
from scipy.io.wavfile import write as wavwrite


# Import the stimulus module for sweep generation and deconvolution

import stimulus
try:
    from scipy.signal.windows import tukey
    stimulus.signal.tukey = tukey
except ImportError:
    pass

# --- Patch: Import utils for audio play/record ---
import utils



def list_audio_outputs():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("Available audio output devices:")
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                print(f"{i}: {device['name']}")
    except Exception as e:
        print("Error listing audio output devices:", e)


# Audio comes out of a binaural renderer through a virtual audio device (output) called BlackHole 2ch
# Use sounddevice to grab audio from the virtual microphone with the same name.
def setup_audio_routing():
    # Set up audio sources for left and right channels (BlackHole 2ch)
    input_device_name_string = "BlackHole 2ch"
    input_devices = sd.query_devices()
    # set input_device to element in list with name input_device_name_string
    input_device = input_devices[[i for i, device in enumerate(input_devices) if device['name'] == input_device_name_string][0]]
    print(input_device)

setup_audio_routing()

def xtc_lab_processor():
    """
      (1) LeftEar=Impulse, RightEar=0
      (2) LeftEar=0,       RightEar=Impulse
    """

    app = QtWidgets.QApplication([])

    # ----------------------------------------------------
    # Geometry & Setup
    # ----------------------------------------------------
    distance_l = 1.5
    distance_r = 1.5
    left_az_deg = -25.6
    right_az_deg = 25.6

    head_position = np.array([1.0, 1.0])
    scale_factor = 100.0
    c = 343.0  # Speed of sound

    xL = head_position[0] + distance_l * sin(radians(left_az_deg))
    yL = head_position[1] + distance_l * cos(radians(left_az_deg))
    xR = head_position[0] + distance_r * sin(radians(right_az_deg))
    yR = head_position[1] + distance_r * cos(radians(right_az_deg))

    # This is structured as follows:
    #   [[xL, yL],  # Left Speaker
    #    [xR, yR]]  # Right Speaker

    original_speaker_positions = np.array([[xL, yL],
                                           [xR, yR]])

    # ----------------------------------------------------
    # MainWindow
    # ----------------------------------------------------
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            import json
            try:
                with open("xtc_lab_config.json", "r") as f:
                    self.config = json.load(f)
            except Exception:
                self.config = {
                    "binaural_device": "...",
                    "binaural_left_channel": 0,
                    "binaural_right_channel": 1,
                    "measurement_device": "...",
                    "measurement_left_channel": 0,
                    "measurement_right_channel": 1,
                    "reference_channel": 4,
                    "playback_device": "...",
                    "playback_left_channel": 0,
                    "playback_right_channel": 1,
                    "hrtf_file": "filename.sofa"
                }
            self.filter_lock = threading.RLock()
            self.samplerate = None

            self.setWindowTitle("XTC Lab Processor - Real-Time Mode")
            self.resize(1920, 1080)

            # Instance variables
            self.hll = None
            self.hlr = None
            self.hrl = None
            self.hrr = None

            self.FLL = None
            self.FLR = None
            self.FRL = None
            self.FRR = None

            self.head_rotation = 0
            self.head_rotation_offset = 0

            self.fll = None
            self.flr = None
            self.frl = None
            self.frr = None

            self.real_hrirs_mode = True
            
            self.LSPKTF = None
            self.RSPKTF = None

            self.current_sofa_file = None
            self.regularization = 0.01
            if os.path.exists("regularization_profile.json"):
                with open("regularization_profile.json", "r") as f:
                    rdata = json.load(f)
                    self.reg_freqs = np.array(rdata["freqs"])
                    self.reg_values = np.array(rdata["values"])
                    self.regularization_interp = interp1d(
                        self.reg_freqs, self.reg_values,
                        bounds_error=False, fill_value=(self.reg_values[0], self.reg_values[-1])
                    )

            self.energy_distribution_enabled = False
            # Mixer Dock (Left Side)
            self.mixerDock = QDockWidget("Mixer", self)
            self.mixerDock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )            
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.mixerDock)

            mixer_widget = QtWidgets.QWidget()
            mixer_layout = QtWidgets.QHBoxLayout(mixer_widget)
            mixer_layout.setContentsMargins(0, 5, 0, 5)
            self.mixerDock.setWidget(mixer_widget)

            # Define channels
            channel_labels = ["IL", "IR", "OL", "OR"]
            self.meter_bars = []
            self.faders = []

            for label in channel_labels:
                channel_widget = QtWidgets.QWidget()
                channel_layout = QtWidgets.QVBoxLayout(channel_widget)
                
                label_widget = QtWidgets.QLabel(label)
                label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                meter_row = QtWidgets.QHBoxLayout()
                meter = level_meter.LevelMeter()
                meter.set_level(0.0)
                
                # Create vertical layout for dB scale
                # db_layout = QtWidgets.QVBoxLayout()
                # db_layout.setSpacing(0)
                # db_layout.setContentsMargins(0, 0, 0, 0)
                
                # for db in range(0, -66, -6):
                #     tick = QtWidgets.QLabel(f"{db}")
                #     tick.setStyleSheet("font-size: 8px; color: white;")
                #     tick.setAlignment(Qt.AlignmentFlag.AlignRight)
                #     db_layout.addWidget(tick, 1)
                
                # db_container = QtWidgets.QWidget()
                # db_container.setLayout(db_layout)
                
                # meter_row.addWidget(db_container)
                meter_row.addWidget(meter)
                
                fader = QtWidgets.QSlider(Qt.Orientation.Vertical)
                fader.setRange(0, 100)
                fader.setValue(80)
                fader.setFixedHeight(200)
                
                channel_layout.addWidget(label_widget)
                meter_container = QtWidgets.QWidget()
                meter_container.setLayout(meter_row)
                channel_layout.addWidget(meter_container, stretch=3, alignment=Qt.AlignmentFlag.AlignHCenter)
                channel_layout.addWidget(fader, stretch=1, alignment=Qt.AlignmentFlag.AlignHCenter)
                # set maximum width of each channel strip
                channel_widget.setMaximumWidth(50)
                mixer_layout.addWidget(channel_widget)
                self.meter_bars.append(meter)
                self.faders.append(fader)
                # if label == "Mic" or label == "IR":
                #     separator = QtWidgets.QFrame()
                #     separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
                #     separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
                #     mixer_layout.addWidget(separator)

            # FFT Dock
            self.fftDock = QDockWidget("Crosstalk FFTs", self)
            features = (QDockWidget.DockWidgetFeature.DockWidgetMovable |
                        QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.fftDock.setFeatures(features)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.fftDock)

            fft_container = QtWidgets.QWidget()
            fft_layout = QtWidgets.QGridLayout(fft_container)
            self.fftDock.setWidget(fft_container)

            self.tf_plots = []
            self.tf_curves = []

            titles = [
                "TF: Left Ear",
                "TF: Right Ear",
                "Filter IR: f11",
                "Filter IR: f12",
                "Filter IR: f21",
                "Filter IR: f22"
            ]
            colors = ['c', 'g']
            for i, title in enumerate(titles):
                pw = pg.PlotWidget(title=title)
                # set the plots to have their grid enabled by default
                pw.showGrid(x=True, y=True)
                if i < 2:
                    pw.setLabel("bottom", "Frequency (Hz)")
                    pw.setLabel("left", "Magnitude (dB)")
                else:
                    pw.setLabel("bottom", "Time (s)")
                    pw.setLabel("left", "Amplitude")
                
                if i < 2:  # TF plots
                    pw.addLegend(offset=(10, 10))
                    curve1 = pw.plot(pen=colors[0], name="Ipsilateral")
                    curve2 = pw.plot(pen=colors[1], name="Contralateral")
                    self.tf_curves.append((curve1, curve2))
                else:      # IR plots
                    curve = pw.plot(pen=colors[0])
                    self.tf_curves.append((curve,))
                
                self.tf_plots.append(pw)
                fft_layout.addWidget(pw, i // 2, i % 2)
                
            self.plot = DraggablePlot(title="Geometric View")
            self.plot.setAspectLocked(True)
            # set minimum width of the plotf
            self.plot.setMinimumWidth(500)

            # set the plot to have it's grid enabled by default
            self.plot.showGrid(x=True, y=True)

            self.energy_item = pg.ImageItem()
            self.energy_item.setZValue(-100)
            self.energy_item.setVisible(False)
            self.plot.addItem(self.energy_item)

            spk_x = original_speaker_positions[:, 0] * scale_factor
            spk_y = original_speaker_positions[:, 1] * scale_factor
            self.speaker_marker = pg.ScatterPlotItem(
                pen=None, brush=pg.mkBrush(0, 0, 255, 150), size=15
            )
            self.speaker_marker.setData(spk_x, spk_y)
            self.plot.addItem(self.speaker_marker)

            # --- Hover panel for speaker-filter FFT chains as floating widget ---
            from PyQt6.QtGui import QCursor
            from PyQt6.QtCore import QPoint
            self.hover_panel = QtWidgets.QWidget(self, flags=Qt.WindowType.ToolTip)
            grid = QtWidgets.QGridLayout(self.hover_panel)
            # reduce padding around the grid
            grid.setContentsMargins(5, 5, 5, 5)
            # Real-time input FFTs
            self.hover_plot_input_L = pg.PlotWidget(title="Input FFT L", showGrid=True, showAxis=False, pen=pg.mkPen('b', width=2))
            self.hover_plot_input_R = pg.PlotWidget(title="Input FFT R", showGrid=True, showAxis=False, pen=pg.mkPen('r', width=2))
            # Filter IR FFTs placeholders
            self.hover_plot_filter1 = pg.PlotWidget(title="Filter FFT 1", showGrid=True, showAxis=False, pen=pg.mkPen('g', width=2))
            self.hover_plot_filter2 = pg.PlotWidget(title="Filter FFT 2", showGrid=True, showAxis=False, pen=pg.mkPen('g', width=2))
            # Output FFT placeholder
            self.hover_plot_output = pg.PlotWidget(title="Output FFT", showGrid=True, showAxis=False)
            # Scale down all hover plots
            for pw in (self.hover_plot_input_L, self.hover_plot_input_R,
                       self.hover_plot_filter1, self.hover_plot_filter2,
                       self.hover_plot_output):
                pw.setFixedSize(150, 100)
                pw.showGrid(x=True, y=True)
                pw.showAxis('bottom', show=False)
                pw.showAxis('left', show=False)
            # Layout: two rows, five columns: input, arrow, filter, sum/output
            grid.addWidget(self.hover_plot_input_L, 0, 0)
            grid.addWidget(QtWidgets.QLabel("→"), 0, 1)
            grid.addWidget(self.hover_plot_filter1, 0, 2)
            grid.addWidget(QtWidgets.QLabel("∑"), 0, 3)
            grid.addWidget(self.hover_plot_output, 0, 4)
            grid.addWidget(self.hover_plot_input_R, 1, 0)
            grid.addWidget(QtWidgets.QLabel("→"), 1, 1)
            grid.addWidget(self.hover_plot_filter2, 1, 2)
            # Empty cells at (1,3) and (1,4) if desired
            self.hover_panel.setLayout(grid)
            self.hover_panel.hide()
            # Connect mouse moves for hover detection
            self.plot.scene().sigMouseMoved.connect(self.on_plot_mouse_moved)
            self.hovered_speaker = None
            self.hover_timer = QtCore.QTimer(self)
            self.hover_timer.setInterval(100)        # refresh every 100 ms
            self.hover_timer.timeout.connect(self.update_hover_fft_display)
            self.hover_timer.start()

            self.head_circle = pg.ScatterPlotItem(
                pen=pg.mkPen('r', width=2), brush=None, size=30
            )

            # Add labels for each speaker
            self.speaker_labels = []
            for i, pos in enumerate(original_speaker_positions):
                label = pg.TextItem(f"Speaker {'L' if i == 0 else 'R'}", anchor=(0.5, -0.5))
                label.setPos(pos[0] * scale_factor, pos[1] * scale_factor)
                self.speaker_labels.append(label)
                self.plot.addItem(label)

            self.head_circle.setZValue(2)
            self.plot.addItem(self.head_circle)

            self.ear_dots = pg.ScatterPlotItem(
                pen=None, brush=pg.mkBrush(255, 0, 0, 150), size=10
            )
            self.ear_dots.setZValue(3)
            self.plot.addItem(self.ear_dots)

            # Speaker->Ear lines
            self.speaker_ear_lines = []
            self.speaker_ear_labels = []
            line_pen = pg.mkPen(color="gray", width=1, dash=[2, 4])
            for _ in range(4):
                line_item = pg.PlotCurveItem(pen=line_pen)
                self.speaker_ear_lines.append(line_item)
                self.plot.addItem(line_item)
                # Angle indicators
                self.angle_labels = []
                for i, pos in enumerate(original_speaker_positions):
                    angle = np.degrees(np.arctan2(pos[0] - head_position[0], pos[1] - head_position[1]))
                    angle_label = pg.TextItem(f"{angle:.1f}°", anchor=(0.5, -0.5))
                    angle_label.setPos((head_position[0]) * scale_factor, (head_position[1] - 20) * scale_factor)
                    angle_label = pg.TextItem(f"{angle:.1f}°", anchor=(0.5, -0.5))
                    angle_label.setPos((pos[0] - 10) * scale_factor, (pos[1] + 10) * scale_factor)
                    self.angle_labels.append(angle_label)
                    self.plot.addItem(angle_label)

            # Layout
            center_widget = QtWidgets.QWidget()
            center_layout = QtWidgets.QVBoxLayout(center_widget)
            self.setCentralWidget(center_widget)
            center_layout.addWidget(self.plot)

            # slider for head rotation
            self.head_rotation_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            self.head_rotation_slider.setRange(-180, 180)
            center_layout.addWidget(self.head_rotation_slider)
            self.head_rotation_slider.setValue(0)
            self.head_rotation_slider.valueChanged.connect(self.update_head_rotation)

            reset_button = QtWidgets.QPushButton("Reset Head Position")
            reset_button.clicked.connect(self.reset_head)
            center_layout.addWidget(reset_button)

            settings_button = QtWidgets.QPushButton("Settings")
            settings_button.clicked.connect(self.open_settings_menu)
            center_layout.addWidget(settings_button)
            grab_hrirs = QtWidgets.QPushButton("Measure and Invert")
            grab_hrirs.clicked.connect(self.grab_hrirs)
            center_layout.addWidget(grab_hrirs)

            inspect_hrir_button = QtWidgets.QPushButton("Inspect HRIRs")
            inspect_hrir_button.clicked.connect(self.open_hrir_inspection_modal)
            center_layout.addWidget(inspect_hrir_button)

            zero_head_angle = QtWidgets.QPushButton("Zero Head Tracker")
            zero_head_angle.clicked.connect(self.calibrate_tracker)
            center_layout.addWidget(zero_head_angle)

            self.filter_design_dropdown = QComboBox()
            self.filter_design_dropdown.addItem("Adjugate Inversion")
            self.filter_design_dropdown.addItem("Kirkeby Nelson Constant Reg")
            self.filter_design_dropdown.addItem("Kirkeby Nelson Smart")

            self.filter_design_dropdown.currentIndexChanged.connect(self.reload_filters)
            center_layout.addWidget(self.filter_design_dropdown)

            self.bypass_checkbox = QtWidgets.QCheckBox("Bypass Filters")
            self.bypass_checkbox.setChecked(True)
            self.bypass_checkbox.stateChanged.connect(self.toggle_bypass)
            center_layout.addWidget(self.bypass_checkbox)

            # Head Tracker Checkbox
            self.headtrack_checkbox = QtWidgets.QCheckBox("Enable Head Tracker")
            self.headtrack_checkbox.stateChanged.connect(self.toggle_headtracker)
            center_layout.addWidget(self.headtrack_checkbox)

            # Head tracking state
            self.headtracker_enabled = False
            self.head_yaw = 0.0

            self.energy_checkbox = QtWidgets.QCheckBox("Show Energy Distribution")
            self.energy_checkbox.stateChanged.connect(self.toggle_energy_distribution)
            center_layout.addWidget(self.energy_checkbox)

            if self.config["binaural_device"] == "..." or self.config["playback_device"] == "...":
                QtCore.QTimer.singleShot(100, self.open_settings_menu)

            # Set up OSC for head tracking
            self.osc_dispatcher = dispatcher.Dispatcher()
            self.osc_dispatcher.set_default_handler(self.osc_handler)
            self.osc_server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 8000), self.osc_dispatcher)
            self.osc_thread = threading.Thread(target=self.osc_server.serve_forever, daemon=True)
            self.osc_thread.start()

            # Timer to update head rotation from tracker
            self.headtrack_timer = QTimer(self)
            self.headtrack_timer.setInterval(50)  # 20 Hz update
            self.headtrack_timer.timeout.connect(self.update_headtrack)
            self.headtrack_timer.start()

            self.auto_range_topdown_plot()

            self.load_default_filters()
            # Start the audio engine at launch
            self.update_audio_engine()
            self.update_plot()
            
        def update_hover_fft_display(self):
            """Periodically refresh the hover-panel FFT plots if visible."""
            if not self.hover_panel.isVisible() or self.hovered_speaker is None:
                return

            # --- Real-time input FFTs from AudioEngine ---
            block = getattr(self.audio_engine, "latest_binaural_block", None)
            if block is not None:
                data_L = block[:, 0]
                data_R = block[:, 1]
            else:
                # fallback to zeros
                data_L = np.zeros(512)
                data_R = np.zeros(512)
            N = len(data_L)
            freqs = np.fft.rfftfreq(N, d=1.0 / self.samplerate)
            mag_L = np.abs(np.fft.rfft(data_L))
            mag_R = np.abs(np.fft.rfft(data_R))
            # Update input FFT plots
            self.hover_plot_input_L.clear()
            self.hover_plot_input_L.plot(freqs, 20 * np.log10(mag_L + 1e-12))
            self.hover_plot_input_R.clear()
            self.hover_plot_input_R.plot(freqs, 20 * np.log10(mag_R + 1e-12))

            # --- Real-time output FFT for hovered speaker ---
            out_block = getattr(self.audio_engine, "latest_output_block", None)
            if out_block is not None:
                # select channel: 0 for left speaker, 1 for right speaker
                chan = self.hovered_speaker
                data_out = out_block[:, chan]
            else:
                data_out = np.zeros(N)
            mag_out = np.abs(np.fft.rfft(data_out))
            self.hover_plot_output.clear()
            self.hover_plot_output.plot(freqs, 20 * np.log10(mag_out + 1e-12))

        def on_plot_mouse_moved(self, scene_pos):
            """
            Show or hide the hover panel when cursor is over a speaker dot.
            """
            from PyQt6.QtGui import QCursor
            from PyQt6.QtCore import QPoint
            # Map scene coordinate to data coordinates (un-scaled)
            vb = self.plot.getViewBox()
            mp = vb.mapSceneToView(scene_pos)
            x = mp.x() / scale_factor
            y = mp.y() / scale_factor

            # Check proximity to each speaker
            for i, sp in enumerate(original_speaker_positions):
                if np.hypot(x - sp[0], y - sp[1]) < 0.1:
                    # Position the floating panel just above the cursor
                    global_pos = QCursor.pos()
                    self.hover_panel.move(global_pos + QPoint(10, -self.hover_panel.height() - 10))
                    # Remember which speaker is hovered
                    self.hovered_speaker = i
                    # Compute and update filter FFTs once on hover
                    if i == 0:
                        arr1, arr2 = self.fll, self.frl
                    else:
                        arr1, arr2 = self.flr, self.frr
                    freqs = np.fft.rfftfreq(len(arr1), d=1.0 / self.samplerate)
                    mag1 = np.abs(np.fft.rfft(arr1))
                    mag2 = np.abs(np.fft.rfft(arr2))
                    self.hover_plot_filter1.clear()
                    self.hover_plot_filter1.plot(freqs, 20 * np.log10(mag1 + 1e-12))
                    self.hover_plot_filter2.clear()
                    self.hover_plot_filter2.plot(freqs, 20 * np.log10(mag2 + 1e-12))
                    # Show panel and return
                    self.hover_panel.show()
                    return

            # Not hovering any speaker
            self.hovered_speaker = None
            self.hover_panel.hide()

        def update_audio_engine(self):
            if hasattr(self, "audio_engine"):
                # Stop and close existing streams before recreating
                try:
                    self.audio_engine.close()
                except Exception:
                    pass
            with self.filter_lock:
                # If frequency domain filters are available, convert to time domain for AudioEngine
                if self.FLL is not None:
                    # Recover full time-domain filters (preserving phase and delay)
                    f11 = np.real(np.fft.ifft(self.FLL))
                    f12 = np.real(np.fft.ifft(self.FLR))
                    f21 = np.real(np.fft.ifft(self.FRL))
                    f22 = np.real(np.fft.ifft(self.FRR))

                    f11 = np.roll(f11, -len(f11) // 2)  # Center the impulse response
                    f12 = np.roll(f12, -len(f11) // 2)
                    f21 = np.roll(f21, -len(f11) // 2)
                    f22 = np.roll(f22, -len(f11) // 2)
                    # window around the centre to achieve length of 8,192
                    # index to centre point - 8192/2
                    center_index = len(f11) // 2
                    f11 = f11[center_index - 1024:center_index + 1024]
                    f12 = f12[center_index - 1024:center_index + 1024]
                    f21 = f21[center_index - 1024:center_index + 1024]
                    f22 = f22[center_index - 1024:center_index + 1024]

                    print("Passing time-domain filters to AudioEngine")
                else:
                    f11 = self.fll
                    f12 = self.flr
                    f21 = self.frl
                    f22 = self.frr
                self.audio_engine = AudioEngine(
                    config=self.config,
                    meter_callback=self.update_meters,
                    f11=f11,
                    f12=f12,
                    f21=f21,
                    f22=f22,
                    bypass=self.bypass_checkbox.isChecked()
                )

        def toggle_bypass(self, state):
            """Toggle filter bypass without restarting audio streams."""
            checked = self.bypass_checkbox.isChecked()
            if hasattr(self, "audio_engine"):
                self.audio_engine.set_bypass(checked)
            
        def open_settings_menu(self):
            settings_dialog = QtWidgets.QDialog(self)
            settings_dialog.setWindowTitle("Settings")
            settings_dialog.setModal(True)
            layout = QtWidgets.QVBoxLayout(settings_dialog)

            devices = sd.query_devices()

            # Device selectors
            binaural_device_combo = QComboBox()
            measurement_device_combo = QComboBox()
            playback_device_combo = QComboBox()

            for combo in [binaural_device_combo, measurement_device_combo, playback_device_combo]:
                combo.addItem("")

            for device in devices:
                if device['max_input_channels'] > 0:
                    binaural_device_combo.addItem(device['name'])
                    measurement_device_combo.addItem(device['name'])
                if device['max_output_channels'] > 0:
                    playback_device_combo.addItem(device['name'])

            layout.addWidget(QLabel("Select Binaural Device:"))
            layout.addWidget(binaural_device_combo)

            layout.addWidget(QLabel("Select Measurement Interface:"))
            layout.addWidget(measurement_device_combo)

            layout.addWidget(QLabel("Select Playback Interface:"))
            layout.addWidget(playback_device_combo)

            # Channel selectors
            binaural_left_combo = QComboBox()
            binaural_right_combo = QComboBox()
            reference_mic_combo = QComboBox()
            ear_mic_left_combo = QComboBox()
            ear_mic_right_combo = QComboBox()
            playback_left_combo = QComboBox()
            playback_right_combo = QComboBox()

            layout.addWidget(QLabel("Select Binaural Left Channel:"))
            layout.addWidget(binaural_left_combo)

            layout.addWidget(QLabel("Select Binaural Right Channel:"))
            layout.addWidget(binaural_right_combo)

            layout.addWidget(QLabel("Select Left Ear Microphone Channel:"))
            layout.addWidget(ear_mic_left_combo)

            layout.addWidget(QLabel("Select Right Ear Microphone Channel:"))
            layout.addWidget(ear_mic_right_combo)

            layout.addWidget(QLabel("Select Reference Microphone Channel:"))
            layout.addWidget(reference_mic_combo)

            layout.addWidget(QLabel("Select Playback Left Channel:"))
            layout.addWidget(playback_left_combo)

            layout.addWidget(QLabel("Select Playback Right Channel:"))
            layout.addWidget(playback_right_combo)

            # HRTF file selection
            hrtf_combo = QComboBox()
            sofa_files = [f for f in os.listdir(".") if f.endswith(".sofa")]
            hrtf_combo.addItem("")
            for f in sofa_files:
                hrtf_combo.addItem(f)

            layout.addWidget(QLabel("Select HRTF File:"))
            layout.addWidget(hrtf_combo)

            # Helper to populate channels
            def populate_channels(device_name, channel_combo_boxes, is_input):
                channel_combo_boxes = channel_combo_boxes if isinstance(channel_combo_boxes, list) else [channel_combo_boxes]
                for c in channel_combo_boxes:
                    c.clear()
                if not device_name:
                    return

                index = next((i for i, d in enumerate(devices) if d['name'] == device_name), None)
                if index is None:
                    return

                channels = devices[index]['max_input_channels'] if is_input else devices[index]['max_output_channels']
                for c in channel_combo_boxes:
                    for ch in range(channels):
                        c.addItem(f"Channel {ch + 1}", ch)

            # Connect combo boxes to populate respective channels
            binaural_device_combo.currentTextChanged.connect(lambda name: populate_channels(name, [binaural_left_combo, binaural_right_combo], is_input=True))
            measurement_device_combo.currentTextChanged.connect(lambda name: populate_channels(name, [ear_mic_left_combo, ear_mic_right_combo, reference_mic_combo], is_input=True))
            playback_device_combo.currentTextChanged.connect(lambda name: populate_channels(name, [playback_left_combo, playback_right_combo], is_input=False))

            # Save button
            save_button = QPushButton("Save")
            layout.addWidget(save_button)

            def save_config():
                self.config = {
                    "binaural_device": binaural_device_combo.currentText(),
                    "binaural_left_channel": binaural_left_combo.currentData(),
                    "binaural_right_channel": binaural_right_combo.currentData(),
                    "measurement_device": measurement_device_combo.currentText(),
                    "measurement_left_channel": ear_mic_left_combo.currentData(),
                    "measurement_right_channel": ear_mic_right_combo.currentData(),
                    "reference_channel": reference_mic_combo.currentData(),
                    "playback_device": playback_device_combo.currentText(),
                    "playback_left_channel": playback_left_combo.currentData(),
                    "playback_right_channel": playback_right_combo.currentData(),
                    "hrtf_file": hrtf_combo.currentText()
                }
                self.current_sofa_file = hrtf_combo.currentText()
                
                with open("xtc_lab_config.json", "w") as f:
                    import json
                    json.dump(self.config, f)
                self.reload_filters()
                self.update_audio_engine()
                self.update_plot()
                settings_dialog.accept()



            save_button.clicked.connect(save_config)
            
            # Restore saved config values
            binaural_device_combo.setCurrentText(self.config["binaural_device"])
            measurement_device_combo.setCurrentText(self.config["measurement_device"])
            playback_device_combo.setCurrentText(self.config["playback_device"])

            populate_channels(self.config["binaural_device"], [binaural_left_combo, binaural_right_combo], is_input=True)
            populate_channels(self.config["measurement_device"], [ear_mic_left_combo, ear_mic_right_combo, reference_mic_combo], is_input=True)
            populate_channels(self.config["playback_device"], [playback_left_combo, playback_right_combo], is_input=False)

            binaural_left_combo.setCurrentIndex(self.config.get("binaural_left_channel", 0))
            binaural_right_combo.setCurrentIndex(self.config.get("binaural_right_channel", 0))
            ear_mic_left_combo.setCurrentIndex(self.config.get("measurement_left_channel", 0))
            ear_mic_right_combo.setCurrentIndex(self.config.get("measurement_right_channel", 0))
            reference_mic_combo.setCurrentIndex(self.config.get("reference_channel", 0))
            playback_left_combo.setCurrentIndex(self.config.get("playback_left_channel", 0))
            playback_right_combo.setCurrentIndex(self.config.get("playback_right_channel", 0))
            hrtf_combo.setCurrentText(self.config.get("hrtf_file", ""))

            settings_dialog.exec()

        def open_geometry_menu(self):
            geometry_dialog = QtWidgets.QDialog(self)
            geometry_dialog.setWindowTitle("Geometry Setup")
            geometry_dialog.setModal(True)

            layout = QtWidgets.QVBoxLayout(geometry_dialog)

            spacing_label = QLabel("Speaker Spacing (cm):")
            layout.addWidget(spacing_label)

            spacing_input = QtWidgets.QDoubleSpinBox()
            spacing_input.setRange(0, 500)
            spacing_input.setValue(self.config.get('speaker_spacing_cm', 150))
            spacing_input.setSuffix(" cm")
            layout.addWidget(spacing_input)

            save_button = QPushButton("Save")
            layout.addWidget(save_button)

            def save_geometry():
                spacing_cm = spacing_input.value()
                self.config['speaker_spacing_cm'] = spacing_cm
                with open("xtc_lab_config.json", "w") as f:
                    import json
                    json.dump(self.config, f)
                self.reload_filters()
                self.update_plot()
                geometry_dialog.accept()

            save_button.clicked.connect(save_geometry)

            geometry_dialog.exec()
        
        def update_head_rotation(self):
            self.head_rotation = self.head_rotation_slider.value()
            # Update the head circle position
            self.reload_filters()
            self.update_plot()

        def update_meters(self, levels):
            for i, level in enumerate(levels[:len(self.meter_bars)]):
                self.meter_bars[i].set_level(level)
                
        def get_azimuth_pair(self):
            """
            Get the azimuth pair for the current head position.
            """
            # Compute source azimuths relative to the (possibly rotated) head
            source_azimuths = []
            for pos in original_speaker_positions:
                dx = pos[0] - head_position[0]
                dy = pos[1] - head_position[1]
                # arctan2(dx, dy) yields angle from +Y axis
                base_angle = np.degrees(np.arctan2(dx, dy))
                # Offset by head rotation and wrap into [0,360)
                az = (base_angle + self.head_rotation) % 360
                source_azimuths.append(az)
            source_azimuth_L, source_azimuth_R = source_azimuths
            print(f"Source azimuths: L={source_azimuth_L:.1f}°, R={source_azimuth_R:.1f}°")
            return source_azimuth_L, source_azimuth_R


        def reload_filters(self, format='FrequencyDomain'):
            if not self.real_hrirs_mode:
                try:
                    left_az_deg, right_az_deg = self.get_azimuth_pair()  
                    # ear, speaker              
                    HRIR_LL, HRIR_RL, sample_rate_l = xtc.extract_hrirs_sam(self.current_sofa_file, left_az_deg, show_plots=False, attempt_interpolate=True) # Left speaker left ear, Left speaker right ear
                    HRIR_LR, HRIR_RR, sample_rate_r = xtc.extract_hrirs_sam(self.current_sofa_file, right_az_deg, show_plots=False, attempt_interpolate=True) 
                    self.hll = HRIR_LL
                    self.hlr = HRIR_LR
                    self.hrl = HRIR_RL
                    self.hrr = HRIR_RR

                    assert sample_rate_l == sample_rate_r, "Sample rates do not match!"
                    self.samplerate = sample_rate_l[0]
                    H_LL, H_LR, H_RL, H_RR = HRIR_LL, HRIR_LR, HRIR_RL, HRIR_RR
                    if self.filter_design_dropdown.currentText() == "Kirkeby Nelson Constant Reg":
                        self.FLL, self.FLR, self.FRL, self.FRR = xtc.generate_kn_filter(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, self.head_rotation, filter_length=2048, samplerate=self.samplerate, debug=True)
                    elif self.filter_design_dropdown.currentText() == "Kirkeby Nelson Smart":
                        self.FLL, self.FLR, self.FRL, self.FRR = xtc.generate_kn_filter_smart(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, filter_length=2048, samplerate=self.samplerate, debug=True)
                    else:
                        if format=='TimeDomain':
                            self.fll, self.flr, self.frl, self.frr = xtc.generate_filter(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, filter_length=2048, samplerate=self.samplerate, debug=False, format=format)
                            # Debug: report raw filter peak magnitudes
                            print(f"Filter peaks (raw): fll={np.max(np.abs(self.fll)):.3f}, flr={np.max(np.abs(self.flr)):.3f}, frl={np.max(np.abs(self.frl)):.3f}, frr={np.max(np.abs(self.frr)):.3f}")
                            print("Filters reloaded successfully.")
                            # invalidate filters in omega
                            self.FLL = self.FLR = self.FRL = self.FRR = None
                        else:
                            self.FLL, self.FLR, self.FRL, self.FRR = xtc.generate_filter(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, filter_length=2048, samplerate=self.samplerate, debug=False, format=format)
                            print("Filters reloaded successfully.")
                            # invalidate filters in time
                            self.fll = self.flr = self.frl = self.frr = None

                    self.update_audio_engine()
                except Exception as e:
                    print(f"Error loading default filters: {e}")
                
                
        def load_default_filters(self, format='FrequencyDomain'):
            """
            Load the default P0275 HRIR filters.
            """
            print("Loading default HRTF.")
            self.current_sofa_file = "P0275_FreeFieldComp_48kHz.sofa"

            try:
                left_az_deg, right_az_deg = self.get_azimuth_pair()                
                HRIR_LL, HRIR_RL, sample_rate_l = xtc.extract_hrirs_sam(self.current_sofa_file, left_az_deg) # Left speaker left ear, Left speaker right ear
                HRIR_LR, HRIR_RR, sample_rate_r = xtc.extract_hrirs_sam(self.current_sofa_file, right_az_deg) # Right speaker left ear, Right speaker right ear
                self.hll = HRIR_LL
                self.hlr = HRIR_LR
                self.hrl = HRIR_RL
                self.hrr = HRIR_RR

                assert sample_rate_l == sample_rate_r, "Sample rates do not match!"
                self.samplerate = sample_rate_l[0]
                H_LL, H_LR, H_RL, H_RR = HRIR_LL, HRIR_LR, HRIR_RL, HRIR_RR
                if self.filter_design_dropdown.currentText() == "Kirkeby Nelson":
                    self.FLL, self.FLR, self.FRL, self.FRR = xtc.generate_kn_filter(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, self.head_rotation, filter_length=2048, samplerate=self.samplerate, debug=False)
                else:
                    if format=='TimeDomain':
                        self.fll, self.flr, self.frl, self.frr = xtc.generate_filter(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, filter_length=2048, samplerate=self.samplerate, debug=False, format=format)
                        # Debug: report raw filter peak magnitudes
                        print(f"Filter peaks (raw): fll={np.max(np.abs(self.fll)):.3f}, flr={np.max(np.abs(self.flr)):.3f}, frl={np.max(np.abs(self.frl)):.3f}, frr={np.max(np.abs(self.frr)):.3f}")
                        print("Filters reloaded successfully, format is", format)
                        # invalidate filters in omega
                        self.FLL = self.FLR = self.FRL = self.FRR = None
                    else:
                        self.FLL, self.FLR, self.FRL, self.FRR = xtc.generate_filter(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, filter_length=2048, samplerate=self.samplerate, debug=False, format=format)
                        print("Filters reloaded successfully, format is", format)
                        # invalidate filters in time
                        self.fll = self.flr = self.frl = self.frr = None
            except Exception as e:
                print(f"Error loading default filters: {e}")

        def toggle_energy_distribution(self):
            self.energy_distribution_enabled = self.energy_checkbox.isChecked()
            # Show or hide the energy image
            self.energy_item.setVisible(self.energy_distribution_enabled)
            self.update_plot()

        def auto_range_topdown_plot(self):
            coords_x = list(original_speaker_positions[:,0]) + [head_position[0]]
            coords_y = list(original_speaker_positions[:,1]) + [head_position[1]]

            x_min, x_max = min(coords_x), max(coords_x)
            y_min, y_max = min(coords_y), max(coords_y)

            buffer_ratio = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= x_range*buffer_ratio
            x_max += x_range*buffer_ratio
            y_min -= y_range*buffer_ratio
            y_max += y_range*buffer_ratio

            x_min *= scale_factor
            x_max *= scale_factor
            y_min *= scale_factor
            y_max *= scale_factor
            self.plot.setXRange(x_min, x_max)
            self.plot.setYRange(y_min, y_max)

        def plot_rotational_error(self):
            self.fll = self.flr = self.frl = self.frr = None

            import os
            results_file = "rotational_error.txt"
            # Load previously computed results
            diff_left_list = []
            diff_right_list = []
            angle_list = []
            processed_angles = set()
            if os.path.exists(results_file):
                with open(results_file, 'r') as rf:
                    for line in rf:
                        try:
                            ang_str, dl_str, dr_str = line.strip().split(',')
                            ang = int(ang_str)
                            dl = float(dl_str)
                            dr = float(dr_str)
                        except ValueError:
                            continue
                        processed_angles.add(ang)
                        angle_list.append(ang)
                        diff_left_list.append(dl)
                        diff_right_list.append(dr)

            print("===== PLOTTING ROTATIONAL ERROR =====")
            for angle in range(-180, 175, 1):
                if angle in processed_angles:
                    continue
                print("Processing angle:", angle)
                self.head_rotation = angle
                left_az_deg, right_az_deg = self.get_azimuth_pair()
                print("left_az_deg:", left_az_deg, "right_az_deg:", right_az_deg)
                try:
                    HRIR_LL, HRIR_RL, sample_rate_l = xtc.extract_hrirs_sam(
                        self.current_sofa_file, left_az_deg, show_plots=False, attempt_interpolate=False
                    )
                    HRIR_LR, HRIR_RR, sample_rate_r = xtc.extract_hrirs_sam(
                        self.current_sofa_file, right_az_deg, show_plots=False, attempt_interpolate=False
                    )
                    self.hll, self.hrl, _ = xtc.extract_hrirs_sam(
                        self.current_sofa_file, left_az_deg, show_plots=False, attempt_interpolate=True
                    )
                    self.hlr, self.hrr, _ = xtc.extract_hrirs_sam(
                        self.current_sofa_file, right_az_deg, show_plots=False, attempt_interpolate=True
                    )

                    # self.hll = HRIR_LL
                    # self.hlr = HRIR_LR
                    # self.hrl = HRIR_RL
                    # self.hrr = HRIR_RR
                except Exception as e:
                    print(f"Skipping angle {angle} due to error: {e}")
                    continue
                assert sample_rate_l == sample_rate_r, "Sample rates do not match!"
                self.samplerate = sample_rate_l[0]
                H_LL, H_LR, H_RL, H_RR = HRIR_LL, HRIR_LR, HRIR_RL, HRIR_RR
                self.FLL, self.FLR, self.FRL, self.FRR = xtc.generate_filter(
                    H_LL, H_LR, H_RL, H_RR,
                    original_speaker_positions, head_position,
                    filter_length=2048, samplerate=self.samplerate,
                    debug=False, format='FrequencyDomain'
                )
                print("calling get acoustics data")
                lines = self.get_acoustics_data()
                ipsi_ir_left, contra_ir_left = lines[0]
                ipsi_ir_right, contra_ir_right = lines[1]
                N_left = len(ipsi_ir_left)
                mag_ipsi_left = 20 * np.log10(
                    np.abs(np.fft.fft(ipsi_ir_left)[:N_left // 2]) + 1e-12
                )
                mag_contra_left = 20 * np.log10(
                    np.abs(np.fft.fft(contra_ir_left)[:N_left // 2]) + 1e-12
                )
                N_right = len(ipsi_ir_right)
                mag_ipsi_right = 20 * np.log10(
                    np.abs(np.fft.fft(ipsi_ir_right)[:N_right // 2]) + 1e-12
                )
                mag_contra_right = 20 * np.log10(
                    np.abs(np.fft.fft(contra_ir_right)[:N_right // 2]) + 1e-12
                )
                # Compute average magnitude differences
                diff_left = np.mean(mag_ipsi_left - mag_contra_left)
                diff_right = np.mean(mag_ipsi_right - mag_contra_right)
                # Save result
                angle_list.append(angle)
                diff_left_list.append(diff_left)
                diff_right_list.append(diff_right)
                with open(results_file, 'a') as rf:
                    rf.write(f"{angle},{diff_left},{diff_right}\n")

            # Sort results for plotting
            sorted_data = sorted(zip(angle_list, diff_left_list, diff_right_list), key=lambda x: x[0])
            if sorted_data:
                angles_sorted, diff_left_sorted, diff_right_sorted = zip(*sorted_data)
            else:
                angles_sorted, diff_left_sorted, diff_right_sorted = [], [], []

            # Plot all stored results
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='polar')
            angles_rad = np.deg2rad(angles_sorted)
            ax.plot(angles_rad, diff_left_sorted, 'r-', label='Left Ear ΔdB')
            ax.plot(angles_rad, diff_right_sorted, 'b-', label='Right Ear ΔdB')
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N')
            ax.set_title('Error in Rotational Invariance - Interp\n', va='bottom')
            ax.legend(loc='upper right')
            plt.show()

        def compute_energy_distribution(self):
            # Ensure the energy image is visible
            self.energy_item.setVisible(True)
            
            grid_size = 100
            xvals = np.linspace(-2, 2, grid_size)
            yvals = np.linspace(-2, 2, grid_size)
            energy_map = np.zeros((grid_size, grid_size), dtype=np.float32)

            ir_len = 512
            earL_imp = np.zeros(ir_len)
            earL_imp[0] = 1.0
            earR_imp = np.zeros(ir_len)
            earR_imp[0] = 1.0
            N = (
                max(len(self.fll), len(self.flr), len(self.frl), len(self.frr))
                + max(len(self.hll), len(self.hlr), len(self.hrl), len(self.hrr))
            )
            F_ll = np.fft.fft(self.fll, N)
            F_lr = np.fft.fft(self.flr, N)
            F_rl = np.fft.fft(self.frl, N)
            F_rr = np.fft.fft(self.frr, N)
            d = np.load("hrir_cache.npz")
            self.H_cache_left  = {}
            self.H_cache_right = {}
            for az, HRIR_LL, HRIR_LR, HRIR_RL, HRIR_RR in zip(
                    d["left_az"], d["HRIR_LL"], d["HRIR_LR"], d["HRIR_RL"], d["HRIR_RR"]):
                self.H_cache_left[az]  = (np.fft.fft(HRIR_LL, N), np.fft.fft(HRIR_LR, N))
                self.H_cache_right[az] = (np.fft.fft(HRIR_RL, N), np.fft.fft(HRIR_RR, N))

            for j, yval in enumerate(yvals):
                for i, xval in enumerate(xvals):
                    # Compute azimuth angles for this grid cell
                    left_az_deg  = np.degrees(np.arctan2(yval - head_position[1], xval - head_position[0]))
                    right_az_deg = np.degrees(np.arctan2(yval - head_position[1], xval - head_position[0]))
                    az_L = int(round(left_az_deg/5)*5) % 360
                    az_R = int(round(right_az_deg/5)*5) % 360
                    H_ll, H_lr = self.H_cache_left[az_L]
                    H_rl, H_rr = self.H_cache_right[az_R]

                    # Removed redundant FFTs of HRIRs here to use cached FFTs
                    X_LL = H_ll*F_ll + H_lr*F_rl
                    X_LR = H_ll*F_lr + H_lr*F_rr
                    X_RL = H_rl*F_ll + H_rr*F_rl
                    X_RR = H_rl*F_lr + H_rr*F_rr

                    # Compute magnitudes
                    mag_X_LL = np.abs(X_LL)
                    mag_X_LR = np.abs(X_LR)
                    mag_X_RL = np.abs(X_RL)
                    mag_X_RR = np.abs(X_RR)
                    #print("Magnitudes: LL={}, LR={}, RL={}, RR={}".format(mag_X_LL, mag_X_LR, mag_X_RL, mag_X_RR))
                    # Only use positive frequencies
                    half = N // 2
                    # Average magnitudes over positive bins
                    ipsi_L = np.mean(mag_X_LL[:half])
                    contra_L = np.mean(mag_X_RL[:half])
                    ipsi_R = np.mean(mag_X_RR[:half])
                    contra_R = np.mean(mag_X_LR[:half])
                    # Compute global averages
                    avg_ipsi = (ipsi_L + ipsi_R) / 2
                    avg_contra = (contra_L + contra_R) / 2
                    
                    # take the mean of the avg_ipsi array
                    avg_avg_ipsi = np.mean(avg_ipsi)
                    avg_avg_contra = np.mean(avg_contra)

                    # Store the difference in the energy map
                    energy_map[j, i] = avg_avg_ipsi - avg_avg_contra
                    print("setting pixel to {}".format(energy_map[j, i]))
                    # Update with automatic level scaling and correct orientation
                    display_map = energy_map.T[::-1, :]
                    self.energy_item.setImage(display_map, autoLevels=True)
                    QtWidgets.QApplication.processEvents()

            return xvals, yvals, energy_map
        
        def grab_hrirs(self):
            # Create results directory
            results_dir = os.path.join("results", "reality_inversions_kn_will_0deg")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Helper: Modal countdown dialog
            def show_countdown_dialog(message, seconds):
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Countdown")
                layout = QtWidgets.QVBoxLayout(dlg)
                label = QtWidgets.QLabel(message)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
                countdown_label = QtWidgets.QLabel("")
                countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(countdown_label)
                dlg.setModal(True)
                dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
                dlg.setFixedSize(300, 120)
                def update_countdown(n):
                    countdown_label.setText(f"{n}")
                for i in reversed(range(1, seconds+1)):
                    update_countdown(i)
                    QtWidgets.QApplication.processEvents()
                    time.sleep(1)
                dlg.accept()
                dlg.close()

            # Helper: Modal ready dialog with GO button
            def show_ready_dialog(message):
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Ready")
                layout = QtWidgets.QVBoxLayout(dlg)
                label = QtWidgets.QLabel(message)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
                go_button = QtWidgets.QPushButton("GO")
                layout.addWidget(go_button)
                dlg.setModal(True)
                dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
                dlg.setFixedSize(300, 120)
                go_button.clicked.connect(dlg.accept)
                dlg.exec()
                dlg.close()

            # --- Step 1: Generate Farina sweep and inverse filter using stimulus class ---
            import matplotlib.pyplot as plt
            # Ensure results directory for plots exists
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            # Placeholders for full deconvolved IRs (first sweep)
            full_RIRs_L = None
            full_RIRs_R = None
            samplerate = 48000
            sweep_duration = 5  # Use integer seconds for sweep_duration for compatibility with stimulus class and sounddevice
            sweep_len = int(samplerate * sweep_duration)
            n_sweeps = 1
            hrir_len = 32768
            start_freq = 50
            end_freq = 16000.0

            filter_length_samples = 16384  # Length of the filter to be generated

            testStimulus = stimulus.stimulus('sinesweep', samplerate)
            # amplitude=1.0, single repetition, no silence before/after
            testStimulus.generate(samplerate, sweep_duration, 1.0, 1, 0, 0, (start_freq, end_freq))
            sweep = testStimulus.signal.flatten()
            start_idx = len(sweep) - 1

            # Prepare device indices once
            devices = sd.query_devices()
            in_dev_idx = next(i for i, d in enumerate(devices) if d['name'] == self.config["measurement_device"])
            out_dev_idx = next(i for i, d in enumerate(devices) if d['name'] == self.config["playback_device"])
            sd.default.device = (in_dev_idx, out_dev_idx)

            # Ensure results directory exists for saving raw sweeps
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Step 2: Record and extract 2×2 HRIRs
            show_ready_dialog("Press go to record.")
            # Left speaker to both ears
            H_LL_raws, H_RL_raws = [], []
            for idx in range(n_sweeps):
                out = np.zeros((sweep_len, 2))

                out[:, 0] = sweep
                out[:, 0] *= 10 ** (-10 / 20)

                # Use utils.record to play & record using the configured devices
                rec = utils.record(
                    out,
                    samplerate,
                    [self.config.get("measurement_left_channel", 0) + 1,
                     self.config.get("measurement_right_channel", 1) + 1],
                    [self.config.get("playback_left_channel", 0) + 1,
                     self.config.get("playback_right_channel", 1) + 1]
                )
                # save individual channels
                wavwrite(
                    os.path.join(results_dir, f"sweep_left_L_{idx}.wav"),
                    samplerate,
                    rec[:, 0]
                )
                wavwrite(
                    os.path.join(results_dir, f"sweep_left_R_{idx}.wav"),
                    samplerate,
                    rec[:, 1]
                )
                # Use stimulus deconvolution
                RIRs = testStimulus.deconvolve(rec)
                if idx == 0:
                    full_RIRs_L = RIRs.copy()
                H_LL_raws.append(RIRs[start_idx:start_idx + hrir_len, 0])
                H_RL_raws.append(RIRs[start_idx:start_idx + hrir_len, 1])
            # Right speaker to both ears
            H_LR_raws, H_RR_raws = [], []
            for idx in range(n_sweeps):
                out = np.zeros((sweep_len, 2))
                out[:, 1] = sweep
                out[:, 1] *= 10 ** (-10 / 20)

                # Use utils.record to play & record using the configured devices
                rec = utils.record(
                    out,
                    samplerate,
                    [self.config.get("measurement_left_channel", 0) + 1,
                     self.config.get("measurement_right_channel", 1) + 1],
                    [self.config.get("playback_left_channel", 0) + 1,
                     self.config.get("playback_right_channel", 1) + 1]
                )
                # save individual channels
                wavwrite(
                    os.path.join(results_dir, f"sweep_right_L_{idx}.wav"),
                    samplerate,
                    rec[:, 0]
                )
                wavwrite(
                    os.path.join(results_dir, f"sweep_right_R_{idx}.wav"),
                    samplerate,
                    rec[:, 1]
                )
                RIRs = testStimulus.deconvolve(rec)
                if idx == 0:
                    full_RIRs_R = RIRs.copy()
                H_LR_raws.append(RIRs[start_idx:start_idx + hrir_len, 0])
                H_RR_raws.append(RIRs[start_idx:start_idx + hrir_len, 1])
            # Plot full deconvolved IRs for first sweep
            # (full_RIRs_L and full_RIRs_R are [samples, 2] arrays)
            if full_RIRs_L is not None and full_RIRs_R is not None:
                t_full = np.arange(full_RIRs_L.shape[0]) / samplerate
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                axs[0, 0].plot(t_full, full_RIRs_L[:, 0]); axs[0, 0].set_title('DECONV_IRS_HLL')
                axs[0, 1].plot(t_full, full_RIRs_R[:, 0]); axs[0, 1].set_title('DECONV_IRS_HLR')
                axs[1, 0].plot(t_full, full_RIRs_L[:, 1]); axs[1, 0].set_title('DECONV_IRS_HRL')
                axs[1, 1].plot(t_full, full_RIRs_R[:, 1]); axs[1, 1].set_title('DECONV_IRS_HRR')
                for ax in axs.flatten():
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'untrimmed_hrirs_time_domain.png'))
                # Display the deconvolution IRs interactively
                # fig.show()
                # Do not close immediately so the window stays open
                plt.close(fig)

            # For now, use only the first sweep
            H_LL_raw, H_RL_raw = H_LL_raws[0], H_RL_raws[0]
            H_LR_raw, H_RR_raw = H_LR_raws[0], H_RR_raws[0]


            # Apply this to each HRIR
            # print(f"Bulk delay roll index: {roll_idx}")
            # print("This corresponds to a delay of {:.2f} ms".format(roll_idx / samplerate * 1000))
            # print("which is {} meters at the speed of sound (343 m/s)".format(
            #     roll_idx / samplerate * 343
            # ))

            # Plot raw HRIRs time-domain
            import matplotlib.pyplot as plt
            t_raw = np.arange(hrir_len) / samplerate
            fig_raw, axs_raw = plt.subplots(2, 2, figsize=(10, 8))
            axs_raw[0, 0].plot(t_raw, H_LL_raw); axs_raw[0, 0].set_title('Raw HRIR H_LL')
            axs_raw[0, 1].plot(t_raw, H_LR_raw); axs_raw[0, 1].set_title('Raw HRIR H_LR')
            axs_raw[1, 0].plot(t_raw, H_RL_raw); axs_raw[1, 0].set_title('Raw HRIR H_RL')
            axs_raw[1, 1].plot(t_raw, H_RR_raw); axs_raw[1, 1].set_title('Raw HRIR H_RR')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "hrirs_time_domain.png"))
            plt.close(fig_raw)
            def remove_bulk_delay(ir, threshold_db=-40):
                magnitude_db = 20 * np.log10(np.abs(ir) / np.max(np.abs(ir)) + 1e-12)
                idx_first_above_threshold = np.argmax(magnitude_db >= threshold_db)
                return idx_first_above_threshold
            H_LL_raw_idx = remove_bulk_delay(H_LL_raw)
            H_LR_raw_idx = remove_bulk_delay(H_LR_raw)
            H_RL_raw_idx = remove_bulk_delay(H_RL_raw)
            H_RR_raw_idx = remove_bulk_delay(H_RR_raw)

            roll_idx = min(H_LL_raw_idx, H_LR_raw_idx, H_RL_raw_idx, H_RR_raw_idx)


            H_LL_raw = H_LL_raw[roll_idx:]
            H_LR_raw = H_LR_raw[roll_idx:]
            H_RL_raw = H_RL_raw[roll_idx:]
            H_RR_raw = H_RR_raw[roll_idx:]

            # Normalise raw HRIRs to 0 dB peak
            # find HRIR peak
            peak = max(np.max(np.abs(H_LL_raw)), 
                        np.max(np.abs(H_LR_raw)),
                        np.max(np.abs(H_RL_raw)),
                        np.max(np.abs(H_RR_raw)))
            H_LL_raw /= peak
            H_LR_raw /= peak
            H_RL_raw /= peak
            H_RR_raw /= peak

            hrir_len = len(H_LL_raw)
            assert hrir_len == len(H_LR_raw) == len(H_RL_raw) == len(H_RR_raw), "HRIR lengths do not match!"
            t_raw = np.arange(hrir_len) / samplerate
            fig_raw, axs_raw = plt.subplots(2, 2, figsize=(10, 8))
            axs_raw[0, 0].plot(t_raw, H_LL_raw); axs_raw[0, 0].set_title('HRIR H_LL')
            axs_raw[0, 1].plot(t_raw, H_LR_raw); axs_raw[0, 1].set_title('HRIR H_LR')
            axs_raw[1, 0].plot(t_raw, H_RL_raw); axs_raw[1, 0].set_title('HRIR H_RL')
            axs_raw[1, 1].plot(t_raw, H_RR_raw); axs_raw[1, 1].set_title('HRIR H_RR')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "hrirs_time_domain_normalised.png"))
            plt.close(fig_raw)


            # --- Plot all HRTF FFTs in one 2×2 subplot figure ---
            H_LL_fft = np.fft.fft(H_LL_raw, n=sweep_len)
            H_LR_fft = np.fft.fft(H_LR_raw, n=sweep_len)
            H_RL_fft = np.fft.fft(H_RL_raw, n=sweep_len)
            H_RR_fft = np.fft.fft(H_RR_raw, n=sweep_len)
            freqs = np.fft.fftfreq(sweep_len, d=1.0 / samplerate)
            fig_hrtf, axs_hrtf = plt.subplots(2, 2, figsize=(10, 8))
            axs_hrtf[0, 0].plot(freqs[:sweep_len//2], 20 * np.log10(np.abs(H_LL_fft[:sweep_len//2]) + 1e-9))
            axs_hrtf[0, 0].set_title("H_LL")
            axs_hrtf[0, 1].plot(freqs[:sweep_len//2], 20 * np.log10(np.abs(H_LR_fft[:sweep_len//2]) + 1e-9))
            axs_hrtf[0, 1].set_title("H_LR")
            axs_hrtf[1, 0].plot(freqs[:sweep_len//2], 20 * np.log10(np.abs(H_RL_fft[:sweep_len//2]) + 1e-9))
            axs_hrtf[1, 0].set_title("H_RL")
            axs_hrtf[1, 1].plot(freqs[:sweep_len//2], 20 * np.log10(np.abs(H_RR_fft[:sweep_len//2]) + 1e-9))
            axs_hrtf[1, 1].set_title("H_RR")
            for ax in axs_hrtf.flatten():
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude (dB)")
            plt.tight_layout()
            fig_hrtf.savefig(os.path.join(results_dir, "hrtf_fft_matrix.png"))
            plt.close(fig_hrtf)

            # --- Step 6: Generate frequency-domain filter kernels ---
            # FLL_real, FLR_real, FRL_real, FRR_real = xtc.generate_real_kn_filter(
            #     H_LL_raw, H_LR_raw, H_RL_raw, H_RR_raw, filter_length=sweep_len, samplerate=samplerate, debug=False
            # )

            FLL_real, FLR_real, FRL_real, FRR_real = xtc.generate_real_kn_filter_fd(
                H_LL_fft, H_LR_fft, H_RL_fft, H_RR_fft, filter_length=sweep_len, samplerate=samplerate, debug=False
            )
            # Set filter matrix
            self.FLL = FLL_real 
            self.FLR = FLR_real
            self.FRL = FRL_real
            self.FRR = FRR_real


            H_mat = np.array([[H_LL_raw, H_LR_raw], [H_RL_raw, H_RR_raw]])

            N = len(FLL_real)
            H_f = np.zeros((2, 2, N), dtype=complex)
            for i in range(2):
                for j in range(2):
                    H_f[i, j, :] = fft(H_mat[i][j], n=N)

            HLL_raw_fft = H_f[0, 0, :]
            HLR_raw_fft = H_f[0, 1, :]
            HRL_raw_fft = H_f[1, 0, :]
            HRR_raw_fft = H_f[1, 1, :]


            # --- Plot filter FFT magnitude matrix ---
            # Removed local imports of matplotlib.pyplot and numpy to avoid shadowing global np and plt
            freqs = np.fft.rfftfreq(N, d=1.0 / samplerate)
            fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
            axs1 = axs1.flatten()
            axs1[0].plot(freqs, 20 * np.log10(np.abs(FLL_real[:len(freqs)]) + 1e-12))
            axs1[0].set_title("F_LL")
            axs1[0].set_xlabel("Frequency (Hz)")
            axs1[0].set_ylabel("Magnitude (dB)")
            axs1[1].plot(freqs, 20 * np.log10(np.abs(FLR_real[:len(freqs)]) + 1e-12))
            axs1[1].set_title("F_LR")
            axs1[1].set_xlabel("Frequency (Hz)")
            axs1[1].set_ylabel("Magnitude (dB)")
            axs1[2].plot(freqs, 20 * np.log10(np.abs(FRL_real[:len(freqs)]) + 1e-12))
            axs1[2].set_title("F_RL")
            axs1[2].set_xlabel("Frequency (Hz)")
            axs1[2].set_ylabel("Magnitude (dB)")
            axs1[3].plot(freqs, 20 * np.log10(np.abs(FRR_real[:len(freqs)]) + 1e-12))
            axs1[3].set_title("F_RR")
            axs1[3].set_xlabel("Frequency (Hz)")
            axs1[3].set_ylabel("Magnitude (dB)")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "filter_matrix_fft_kn.png"))
            plt.close(fig1)

            # --- Step 3a: Compute HF products and plot ---
            # H_LL_fft, H_LR_fft, H_RL_fft, H_RR_fft are already computed above
            print("Lengths of FLL, FLR, FRL, FRR:", len(self.FLL), len(self.FLR), len(self.FRL), len(self.FRR))
            print("Lengths of H_LL_fft, H_LR_fft, H_RL_fft, H_RR_fft:",
                  len(H_LL_fft), len(H_LR_fft), len(H_RL_fft), len(H_RR_fft))

            freqs = np.fft.rfftfreq(N, d=1.0 / samplerate)

            X_LL = HLL_raw_fft * self.FLL + HLR_raw_fft * self.FRL
            X_LR = HLL_raw_fft * self.FLR + HLR_raw_fft * self.FRR
            X_RL = HRL_raw_fft * self.FLL + HRR_raw_fft * self.FRL
            X_RR = HRL_raw_fft * self.FLR + HRR_raw_fft * self.FRR

            fig_filt, axs_filt = plt.subplots(2, 2, figsize=(10, 8))
            axs_filt = axs_filt.flatten()
            axs_filt[0].plot(freqs, 20 * np.log10(np.abs(X_LL[:len(freqs)]) + 1e-12))
            axs_filt[0].set_title("X_LL")
            axs_filt[0].set_xlabel("Frequency (Hz)")
            axs_filt[0].set_ylabel("Magnitude (dB)")
            axs_filt[1].plot(freqs, 20 * np.log10(np.abs(X_LR[:len(freqs)]) + 1e-12))
            axs_filt[1].set_title("X_LR")
            axs_filt[1].set_xlabel("Frequency (Hz)")
            axs_filt[1].set_ylabel("Magnitude (dB)")
            axs_filt[2].plot(freqs, 20 * np.log10(np.abs(X_RL[:len(freqs)]) + 1e-12))
            axs_filt[2].set_title("X_RL")
            axs_filt[2].set_xlabel("Frequency (Hz)")
            axs_filt[2].set_ylabel("Magnitude (dB)")
            axs_filt[3].plot(freqs, 20 * np.log10(np.abs(X_RR[:len(freqs)]) + 1e-12))
            axs_filt[3].set_title("X_RR")
            axs_filt[3].set_xlabel("Frequency (Hz)")
            axs_filt[3].set_ylabel("Magnitude (dB)")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'HF_product_matrix_kn.png'))
            plt.close(fig_filt)

            # --- Step 3b: Physical measurement through AudioEngine + binaural device ---
            # Route playback through binaural renderer
            # Update engine to use binaural device and enable filters
            self.update_audio_engine()


            # Save original routing
            # Query device indices
            # Pre-trigger for windowing
            pre_trigger = int(0.004 * samplerate)
            # Measure filtered left-input through engine
            fLL = np.real(np.fft.ifft(self.FLL))
            fLR = np.real(np.fft.ifft(self.FLR))
            fRL = np.real(np.fft.ifft(self.FRL))
            fRR = np.real(np.fft.ifft(self.FRR))


            normalize = np.max(np.abs(fLL)) + np.max(np.abs(fLR)) + np.max(np.abs(fRL)) + np.max(np.abs(fRR))
            fLL /= normalize
            fLR /= normalize
            fRL /= normalize
            fRR /= normalize

            first_idx_ll = remove_bulk_delay(fLL)
            first_idx_lr = remove_bulk_delay(fLR)
            first_idx_rl = remove_bulk_delay(fRL)
            first_idx_rr = remove_bulk_delay(fRR)

            start_idx = min(first_idx_ll, first_idx_lr, first_idx_rl, first_idx_rr)
            

            # plot filter IRs.
            fig_ir, axs_ir = plt.subplots(2, 2, figsize=(10, 8))
            axs_ir[0, 0].plot(fLL)
            axs_ir[0, 0].set_title("Filter fLL Impulse Response")
            axs_ir[0, 1].plot(fLR)
            axs_ir[0, 1].set_title("Filter fLR Impulse Response")
            axs_ir[1, 0].plot(fRL)
            axs_ir[1, 0].set_title("Filter fRL Impulse Response")
            axs_ir[1, 1].plot(fRR)
            axs_ir[1, 1].set_title("Filter fRR Impulse Response")
            plt.tight_layout()
            fig_ir.savefig(os.path.join(results_dir, "filter_impulse_responses_kn.png"))
            plt.close(fig_ir)

            fLL = fLL[start_idx:]
            fLR = fLR[start_idx:]
            fRL = fRL[start_idx:]
            fRR = fRR[start_idx:]

            fig_ir, axs_ir = plt.subplots(2, 2, figsize=(10, 8))
            axs_ir[0, 0].plot(fLL)
            axs_ir[0, 0].set_title("Filter fLL Impulse Response")
            axs_ir[0, 1].plot(fLR)
            axs_ir[0, 1].set_title("Filter fLR Impulse Response")
            axs_ir[1, 0].plot(fRL)
            axs_ir[1, 0].set_title("Filter fRL Impulse Response")
            axs_ir[1, 1].plot(fRR)
            axs_ir[1, 1].set_title("Filter fRR Impulse Response")
            plt.tight_layout()
            fig_ir.savefig(os.path.join(results_dir, "rolled_filter_impulse_responses_kn.png"))
            plt.close(fig_ir)

            filter_len = max(len(fLL), len(fLR), len(fRL), len(fRR))

            # Left channel to both ears, assume s_r = 0
            HF_LL_raws, HF_RL_raws = [], []
            for idx in range(n_sweeps):
                # left ear left channel = fLL * sweep + fLR * sweep
                sweepfft = np.fft.fft(sweep, n=filter_len)
                filtered_left = np.fft.ifft(self.FLL * sweepfft)
                filtered_right = np.fft.ifft(self.FRL * sweepfft)

                conv_len       = len(filtered_left)  # = sweep_len + len(fLL) - 1
                out = np.zeros((conv_len, 2))
                out[:, 0] = filtered_left
                out[:, 1] = filtered_right
                out[:, 0] *= 10 ** (-15 / 20)
                out[:, 1] *= 10 ** (-15 / 20)

                # Use utils.record to play & record using the configured devices
                rec = utils.record(
                    out,
                    samplerate,
                    [self.config.get("measurement_left_channel", 0) + 1,
                     self.config.get("measurement_right_channel", 1) + 1],
                    [self.config.get("playback_left_channel", 0) + 1,
                     self.config.get("playback_right_channel", 1) + 1]
                )
                # save individual channels
                wavwrite(
                    os.path.join(results_dir, f"sweep_left_L_{idx}_FILTERED_kn.wav"),
                    samplerate,
                    rec[:, 0]
                )
                wavwrite(
                    os.path.join(results_dir, f"sweep_left_R_{idx}_FILTERED_kn.wav"),
                    samplerate,
                    rec[:, 1]
                )
                # Use stimulus deconvolution
                rec_LL = rec[:, 0]
                rec_RL = rec[:, 1]
                RIRs = testStimulus.deconvolve(rec)
                if idx == 0:
                    full_RIRs_L = RIRs.copy()
                HF_LL_raws.append(RIRs[start_idx:start_idx + filter_len, 0])
                HF_RL_raws.append(RIRs[start_idx:start_idx + filter_len, 1])
            # Right speaker to both ears


            HF_LR_raws, HF_RR_raws = [], []
            for idx in range(n_sweeps):
                sweepfft = np.fft.fft(sweep, n=filter_len)
                filtered_left  = np.fft.ifft(self.FLR * sweepfft)
                filtered_right = np.fft.ifft(self.FRR * sweepfft)
                conv_len       = len(filtered_left)  # = sweep_len + len(fRL) - 1
                out = np.zeros((conv_len, 2))
                out[:, 0] = filtered_left
                out[:, 1] = filtered_right
                out[:, 0] *= 10 ** (-15 / 20)
                out[:, 1] *= 10 ** (-15 / 20)

                # Use utils.record to play & record using the configured devices
                rec = utils.record(
                    out,
                    samplerate,
                    [self.config.get("measurement_left_channel", 0) + 1,
                     self.config.get("measurement_right_channel", 1) + 1],
                    [self.config.get("playback_left_channel", 0) + 1,
                     self.config.get("playback_right_channel", 1) + 1]
                )
                # save individual channels
                wavwrite(
                    os.path.join(results_dir, f"sweep_right_L_{idx}_FILTERED_kn.wav"),
                    samplerate,
                    rec[:, 0]
                )
                wavwrite(
                    os.path.join(results_dir, f"sweep_right_R_{idx}_FILTERED_kn.wav"),
                    samplerate,
                    rec[:, 1]
                )
                rec_LR = rec[:, 0]
                rec_RR = rec[:, 1]
                RIRs = testStimulus.deconvolve(rec)
                if idx == 0:
                    full_RIRs_R = RIRs.copy()
                HF_LR_raws.append(RIRs[start_idx:start_idx + filter_len, 0])
                HF_RR_raws.append(RIRs[start_idx:start_idx + filter_len, 1])
            

            if full_RIRs_L is not None and full_RIRs_R is not None:
                t_full = np.arange(full_RIRs_L.shape[0]) / samplerate
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                axs[0, 0].plot(t_full, full_RIRs_L[:, 0]); axs[0, 0].set_title('DECONV_IRS_HLL')
                axs[0, 1].plot(t_full, full_RIRs_R[:, 0]); axs[0, 1].set_title('DECONV_IRS_HLR')
                axs[1, 0].plot(t_full, full_RIRs_L[:, 1]); axs[1, 0].set_title('DECONV_IRS_HRL')
                axs[1, 1].plot(t_full, full_RIRs_R[:, 1]); axs[1, 1].set_title('DECONV_IRS_HRR')
                for ax in axs.flatten():
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'untrimmed_hrirs_time_domain_FILTERED_kn.png'))
                # Display the deconvolution IRs interactively
                # fig.show()
                # Do not close immediately so the window stays open
                plt.close(fig)
                # --- Plot FFTs of filtered HRIRs (crosstalk-cancelled) ---
                # Prepare dictionary of the first sweep's crosstalk-cancelled HRIR segments
                # --- Plot FFTs of filtered HRIRs in a single 2×2 subplot figure ---
            Hfs = {
                'Hf_LL': rec_LL,
                'Hf_RL': rec_RL,
                'Hf_LR': rec_LR,
                'Hf_RR': rec_RR,
            }
            freqs = np.fft.rfftfreq(filter_len, d=1.0 / samplerate)
            fig_fft, axs_fft = plt.subplots(2, 2, figsize=(10, 8))
            for ax, (label, data) in zip(axs_fft.flatten(), Hfs.items()):
                H = np.fft.rfft(data, n=filter_len)
                ax.plot(freqs, 20 * np.log10(np.abs(H) + 1e-12))
                ax.set_title(f'Filtered HRIR FFT: {label}')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude (dB)')
            plt.tight_layout()
            fig_fft.savefig(os.path.join(results_dir, 'filtered_hrir_fft_matrix_kn.png'))
            plt.close(fig_fft)
            
            output_dir = "results/xtc_data_will_0deg"
            os.makedirs(output_dir, exist_ok=True)

            # Save original HRIRs
            np.savetxt(os.path.join(output_dir, "H_LL_raw_ir.csv"), H_LL_raw, delimiter=",")
            np.savetxt(os.path.join(output_dir, "H_LR_raw_ir.csv"), H_LR_raw, delimiter=",")
            np.savetxt(os.path.join(output_dir, "H_RL_raw_ir.csv"), H_RL_raw, delimiter=",")
            np.savetxt(os.path.join(output_dir, "H_RR_raw_ir.csv"), H_RR_raw, delimiter=",")

            # Save FFTs of original HRTFs
            np.savetxt(os.path.join(output_dir, "H_LL_raw_fft.csv"), np.abs(H_LL_fft), delimiter=",")
            np.savetxt(os.path.join(output_dir, "H_LR_raw_fft.csv"), np.abs(H_LR_fft), delimiter=",")
            np.savetxt(os.path.join(output_dir, "H_RL_raw_fft.csv"), np.abs(H_RL_fft), delimiter=",")
            np.savetxt(os.path.join(output_dir, "H_RR_raw_fft.csv"), np.abs(H_RR_fft), delimiter=",")

            # Save filter IRs
            np.savetxt(os.path.join(output_dir, "fLL_ir.csv"), fLL, delimiter=",")
            np.savetxt(os.path.join(output_dir, "fLR_ir.csv"), fLR, delimiter=",")
            np.savetxt(os.path.join(output_dir, "fRL_ir.csv"), fRL, delimiter=",")
            np.savetxt(os.path.join(output_dir, "fRR_ir.csv"), fRR, delimiter=",")

            # Save filter FFTs
            np.savetxt(os.path.join(output_dir, "FLL_fft.csv"), np.abs(FLL_real), delimiter=",")
            np.savetxt(os.path.join(output_dir, "FLR_fft.csv"), np.abs(FLR_real), delimiter=",")
            np.savetxt(os.path.join(output_dir, "FRL_fft.csv"), np.abs(FRL_real), delimiter=",")
            np.savetxt(os.path.join(output_dir, "FRR_fft.csv"), np.abs(FRR_real), delimiter=",")

            # Save HF product matrix
            np.savetxt(os.path.join(output_dir, "X_LL_fft.csv"), np.abs(X_LL), delimiter=",")
            np.savetxt(os.path.join(output_dir, "X_LR_fft.csv"), np.abs(X_LR), delimiter=",")
            np.savetxt(os.path.join(output_dir, "X_RL_fft.csv"), np.abs(X_RL), delimiter=",")
            np.savetxt(os.path.join(output_dir, "X_RR_fft.csv"), np.abs(X_RR), delimiter=",")

            # Save second sweep IRs (post-filter measurement)
            np.savetxt(os.path.join(output_dir, "M_LL.csv"), rec_LL, delimiter=",")
            np.savetxt(os.path.join(output_dir, "M_LR.csv"), rec_LR, delimiter=",")
            np.savetxt(os.path.join(output_dir, "M_RL.csv"), rec_RL, delimiter=",")
            np.savetxt(os.path.join(output_dir, "M_RR.csv"), rec_RR, delimiter=",")

            





        def get_acoustics_data(self):
            with self.filter_lock:
                # Compute propagation delays to the head (in samples)
                if self.fll is None or self.flr is None or self.frl is None or self.frr is None:
                    N = self.FLL.shape[0] #number of FFT points.
                else: N = (
                    max(len(self.fll), len(self.flr), len(self.frl), len(self.frr))
                    + max(len(self.hll), len(self.hlr), len(self.hrl), len(self.hrr))
                )
                # We could simulate the impulse, but we don't need to
                # Because it's the same as multiplying by 1.

                H_ll = np.fft.fft(self.hll, N)
                H_lr = np.fft.fft(self.hlr, N)
                H_rl = np.fft.fft(self.hrl, N)
                H_rr = np.fft.fft(self.hrr, N)

                delay_samples_l = int(np.round(np.linalg.norm(original_speaker_positions[0] - head_position) / 343.0 * self.samplerate))
                delay_samples_r = int(np.round(np.linalg.norm(original_speaker_positions[1] - head_position) / 343.0 * self.samplerate))
                if delay_samples_l == delay_samples_r:
                    print("Left and right sources are coherent, no delay applied")
                    tau_L = 0
                    tau_R = 0
                elif delay_samples_l > delay_samples_r:
                    tau_L = 0
                    tau_R = (delay_samples_l - delay_samples_r)/self.samplerate
                    print("applied {} seconds of delay to right".format(tau_R))
                elif delay_samples_r > delay_samples_l:
                    tau_L = (delay_samples_r - delay_samples_l)/self.samplerate
                    tau_R = 0
                    print("applied {} seconds of delay to left".format(tau_L))
                
                if delay_samples_l != 0 or delay_samples_r != 0:
                    freqs = np.fft.fftfreq(N, d=1/self.samplerate)
                    omega = 2 * np.pi * freqs
                
                    exp_L = np.exp(1j * omega * tau_L)    # for the left speaker
                    exp_R = np.exp(1j * omega * tau_R)    # for the right speaker

                    H_ll = H_ll * exp_L
                    H_lr = H_lr * exp_R
                    H_rl = H_rl * exp_L
                    H_rr = H_rr * exp_R


                # match delay applied to make filters causal.
                delay_samples_l = -700
                delay_samples_r = -700
                freqs = np.fft.fftfreq(N, d=1/self.samplerate)
                omega = 2 * np.pi * freqs
                tau_L = delay_samples_l / self.samplerate
                tau_R = delay_samples_r / self.samplerate
                exp_L = np.exp(1j * omega * tau_L)
                exp_R = np.exp(1j * omega * tau_R)

                H_ll *= exp_L
                H_lr *= exp_R
                H_rl *= exp_L
                H_rr *= exp_R

                if self.FLL is not None:
                    F_ll = self.FLL
                    F_lr = self.FLR
                    F_rl = self.FRL
                    F_rr = self.FRR
                else:
                    F_ll = np.fft.fft(self.fll, N)
                    F_lr = np.fft.fft(self.flr, N)
                    F_rl = np.fft.fft(self.frl, N)
                    F_rr = np.fft.fft(self.frr, N)

                freqs = np.fft.fftfreq(N, d=1/self.samplerate)
                omega = 2 * np.pi * freqs

                X_LL = H_ll*F_ll + H_lr*F_rl #left ear left channel
                X_LR = H_ll*F_lr + H_lr*F_rr #left ear right channel
                X_RL = H_rl*F_ll + H_rr*F_rl #right ear left channel
                X_RR = H_rl*F_lr + H_rr*F_rr #right ear right channel

                x_LL = np.fft.ifft(X_LL)
                x_LR = np.fft.ifft(X_LR)
                x_RR = np.fft.ifft(X_RR)
                x_RL = np.fft.ifft(X_RL)

                Left_Ear_Left_Channel = x_LL
                Left_Ear_Right_Channel = x_LR
                Right_Ear_Right_Channel = x_RR
                Right_Ear_Left_Channel = x_RL

                if self.fll is None or self.flr is None or self.frl is None or self.frr is None:
                    self.fll = np.fft.ifft(F_ll).real
                    self.flr = np.fft.ifft(F_lr).real
                    self.frl = np.fft.ifft(F_rl).real
                    self.frr = np.fft.ifft(F_rr).real

                if self.fll is not None:
                    def align_by_peak(ir):
                        peak_index = np.argmax(np.abs(ir))
                        centered = np.roll(ir, len(ir)//2 - peak_index)
                        return centered
                    fll_trimmed = self.fll[:512]
                    flr_trimmed = self.flr[:512]
                    frl_trimmed = self.frl[:512]
                    frr_trimmed = self.frr[:512]

                    f11_centered = align_by_peak(fll_trimmed)
                    f12_centered = align_by_peak(flr_trimmed)
                    f21_centered = align_by_peak(frl_trimmed)
                    f22_centered = align_by_peak(frr_trimmed)

                    t = np.arange(len(f11_centered)) / self.samplerate

                    self.tf_curves[2][0].setData(t, f11_centered)  # f11
                    self.tf_curves[3][0].setData(t, f12_centered)  # f12
                    self.tf_curves[4][0].setData(t, f21_centered)  # f21
                    self.tf_curves[5][0].setData(t, f22_centered)  # f22
            
            #horrible abuse of class variables here
            self.fll = self.flr = self.frl = self.frr = None
            
            return [
                (Left_Ear_Left_Channel, Left_Ear_Right_Channel),   # TF Left Ear
                (Right_Ear_Right_Channel, Right_Ear_Left_Channel),   # TF Right Ear
                (t, f11_centered),
                (t, f12_centered),
                (t, f21_centered),
                (t, f22_centered)
            ]        
        

        def update_plot(self):
            print("Updating plot...")

            # If energy distribution is on, show the image in background
            if self.energy_distribution_enabled:
                xvals, yvals, energy_map = self.compute_energy_distribution()
                x_min = xvals[0]*scale_factor
                y_min = yvals[0]*scale_factor
                width = (xvals[-1]-xvals[0])*scale_factor
                height= (yvals[-1]-yvals[0])*scale_factor
                # Display without transposing so that rows map to y and cols to x
                display_map = energy_map.T[::-1, :]
                self.energy_item.setImage(display_map, autoLevels=True)
                self.energy_item.setRect(pg.QtCore.QRectF(x_min, y_min, width, height))
                self.energy_item.setVisible(True)
            else:
                self.energy_item.setVisible(False)

            # Head / Ears
            hx = head_position[0]*scale_factor
            hy = head_position[1]*scale_factor
            self.head_circle.setData([hx],[hy])

            # Compute rotated ear positions around head center
            theta = np.radians(self.head_rotation)
            ear_offset = 0.15 / 2
            # Offsets for left and right ears (x, y) relative to head center before rotation
            offsets = np.array([[-ear_offset, 0.0], [ear_offset, 0.0]])
            # Rotation matrix for angle theta
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_mat = np.array([[cos_t, -sin_t],
                                [sin_t,  cos_t]])
            # Apply rotation
            rotated = offsets.dot(rot_mat.T)
            # Translate to head position
            ear_positions = head_position + rotated
            # Scale and plot
            xs = ear_positions[:, 0] * scale_factor
            ys = ear_positions[:, 1] * scale_factor
            self.ear_dots.setData(xs, ys)

            # Unpack rotated ear positions for line connections
            L_ear, R_ear = ear_positions
            connections = [
                (original_speaker_positions[0], L_ear),
                (original_speaker_positions[0], R_ear),
                (original_speaker_positions[1], L_ear),
                (original_speaker_positions[1], R_ear),
            ]

            # Update angle labels
            for i, pos in enumerate(original_speaker_positions):
                base_angle = np.degrees(np.arctan2(pos[0] - head_position[0], pos[1] - head_position[1]))
                angle = (base_angle + self.head_rotation) % 360
                if i == 0:
                    self.angle_labels[i].setText(f"H->L: {angle:.1f}°")
                elif i == 1:
                    self.angle_labels[i].setText(f"\nH->R:{angle:.1f}°")
                self.angle_labels[i].setPos(
                    head_position[0] * scale_factor,
                    (head_position[1] - 0.2) * scale_factor
                )
            for i, (spk, ear) in enumerate(connections):
                spk_scaled = spk*scale_factor
                ear_scaled = ear*scale_factor
                self.speaker_ear_lines[i].setData(
                    x=[spk_scaled[0], ear_scaled[0]],
                    y=[spk_scaled[1], ear_scaled[1]]
                )
                dist_m = np.linalg.norm(spk - ear)
                delay_ms = (dist_m/c)*1000
                mid = (spk+ear)/2.0
                mid_scaled = mid*scale_factor
            if not self.bypass_checkbox.isChecked():
                self.reload_filters()
                lines = self.get_acoustics_data()
                for i, data in enumerate(lines):
                    if i == 0:
                        # Left‐ear TFs: ipsi vs contra
                        ipsi_ir, contra_ir = data
                        N = len(ipsi_ir)
                        freqs = np.fft.fftfreq(N, d=1.0/self.samplerate)[:N//2]
                        mag_ipsi = 20*np.log10(np.abs(np.fft.fft(ipsi_ir)[:N//2]) + 1e-12)
                        mag_contra = 20*np.log10(np.abs(np.fft.fft(contra_ir)[:N//2]) + 1e-12)
                        self.tf_curves[0][0].setData(freqs, mag_ipsi)
                        self.tf_curves[0][1].setData(freqs, mag_contra)
                    elif i == 1:
                        # Right‐ear TFs: ipsi vs contra
                        ipsi_ir, contra_ir = data
                        N = len(ipsi_ir)
                        freqs = np.fft.fftfreq(N, d=1.0/self.samplerate)[:N//2]
                        mag_ipsi = 20*np.log10(np.abs(np.fft.fft(ipsi_ir)[:N//2]) + 1e-12)
                        mag_contra = 20*np.log10(np.abs(np.fft.fft(contra_ir)[:N//2]) + 1e-12)
                        self.tf_curves[1][0].setData(freqs, mag_ipsi)
                        self.tf_curves[1][1].setData(freqs, mag_contra)
                    else:
                        # Filter IR plots remain in time domain
                        t, ir = data
                        self.tf_curves[i][0].setData(t, ir)


        def reset_head(self):
            self.head_rotation = 0
            head_position[:] = np.array([1.0, 1.0])
            self.reload_filters()
            self.update_plot()
            
        def open_hrir_inspection_modal(self):
            try:
                self.plot_rotational_error()
                # if self.hll is not None:
                #     data_4 = {
                #         "H_LL": self.hll,
                #         "H_LR": self.hlr,
                #         "H_RL": self.hrl,
                #         "H_RR": self.hrr,
                #         "samplerate": self.samplerate
                #     }
                # else:
                #     data_4 = xtc.extract_hrirs_sam(self.current_sofa_file, left_az=-30.0, right_az=30.0)
                #     print("Opened modal and extracted HRIRs")
                # H_LL, H_LR, H_RL, H_RR = data_4["H_LL"], data_4["H_LR"], data_4["H_RL"], data_4["H_RR"]
                # samplerate = data_4["samplerate"]
                # with self.filter_lock:
                #                     f11, f12, f21, f22 = xtc.generate_filter(
                #                         H_LL, H_LR, H_RL, H_RR,
                #                         head_position=head_position,
                #                         speaker_positions=original_speaker_positions,
                #                         filter_length=8192,
                #                         samplerate=samplerate,
                                    # )
            except Exception as e:
                print(f"Error loading HRIRs: {e}")
                return

            # dialog = QtWidgets.QDialog(self)
            # dialog.setWindowTitle("HRIRs and Inverted Filters")
            # layout = QtWidgets.QVBoxLayout(dialog)

            # plot_widget = pg.GraphicsLayoutWidget()
            # layout.addWidget(plot_widget)

            # def plot_ir_row(title, left_data, right_data):
            #     p = plot_widget.addPlot(title=title)
            #     t = np.arange(len(left_data)) / samplerate
            #     p.plot(t, left_data, pen='r', name="Left")
            #     p.plot(t, right_data, pen='b', name="Right")
            #     p.showGrid(x=True, y=True)
            #     plot_widget.nextRow()

            # plot_ir_row("HRIRs: H_LL and H_RR", H_LL, H_RR)
            # plot_ir_row("HRIRs: H_LR and H_RL", H_LR, H_RL)
            # plot_ir_row("Inverted Filters: f11 and f22", f11, f22)
            # plot_ir_row("Inverted Filters: f12 and f21", f12, f21)

            def open_fft_viewer(H_LL, H_LR, H_RL, H_RR):
                # plot each fft of each IR separately , top grid of 4 H matrix
                # bottom grid of 4 inverted filter matrices f11, f12, f21, f22 etc
                dialog = QtWidgets.QDialog(self)
                dialog.setWindowTitle("FFT Viewer")
                layout = QtWidgets.QVBoxLayout(dialog)

                plot_widget = pg.GraphicsLayoutWidget()
                layout.addWidget(plot_widget)

                def plot_fft_row(title, data_left, data_right):
                    p = plot_widget.addPlot(title=title)
                    fft_len = len(data_left)
                    freq_axis = np.fft.fftfreq(fft_len, d=1.0 / samplerate)[:fft_len // 2]
                    fft_left = 20 * np.log10(np.abs(np.fft.fft(data_left))[:fft_len // 2] + 1e-12)
                    fft_right = 20 * np.log10(np.abs(np.fft.fft(data_right))[:fft_len // 2] + 1e-12)
                    p.plot(freq_axis, fft_left, pen='r', name="Left")
                    p.plot(freq_axis, fft_right, pen='b', name="Right")
                    p.setLabel("bottom", "Frequency (Hz)")
                    p.setLabel("left", "Magnitude (dB)")
                    p.showGrid(x=True, y=True)
                    plot_widget.nextRow()

                # Top row: FFTs of HRIRs
                plot_fft_row("FFT: H_LL", H_LL, H_RR)
                plot_fft_row("FFT: H_LR", H_LR, H_RL)

                # Bottom row: FFTs of inverted filters
                plot_fft_row("FFT: f11", f11, f22)
                plot_fft_row("FFT: f12", f12, f21)

                close_btn = QtWidgets.QPushButton("Close")
                close_btn.clicked.connect(dialog.accept)
                layout.addWidget(close_btn)

                dialog.resize(800, 600)
                dialog.exec()

            view_ffts_btn = QtWidgets.QPushButton("View FFTs")
            view_ffts_btn.clicked.connect(lambda: open_fft_viewer(H_LL, H_LR, H_RL, H_RR))
            layout.addWidget(view_ffts_btn)

            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.resize(800, 600)
            dialog.exec()

        def calibrate_tracker(self):
            self.head_rotation_offset = -self.head_yaw

        def take_acoustics_measurements(self):
            # 1. check / assert that the head tracker is connected and streaming
            return

        def toggle_headtracker(self, state):
            self.headtracker_enabled = self.headtrack_checkbox.isChecked()
            if self.headtracker_enabled:
                print("[HeadTracker] Enabled, starting timer")
                self.headtrack_timer.start()
            else:
                print("[HeadTracker] Disabled, stopping timer")
                self.headtrack_timer.stop()

        def osc_handler(self, address, *args):
            # Expecting yaw as first argument
            if args:
                try:
                    # print(f"OSC message received: {address} with args {args}")
                    if address == "/ypr":
                        # print(f"[HeadTracker] Yaw received: {args[0]}")
                        self.head_yaw = -float(args[0])
                except ValueError:
                    pass

        def update_headtrack(self):
            # print("[HeadTracker] Timer tick")
            if self.headtracker_enabled:
                # print(f"[HeadTracker] Updating plot with yaw: {self.head_yaw}")
                # Update head rotation and redraw top-down plot
                self.head_rotation = self.head_yaw + self.head_rotation_offset
                self.update_plot()

    # ---------------------------------------------
    # DraggablePlot
    # ---------------------------------------------
    class DraggablePlot(pg.PlotWidget):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.dragging = False

        def mousePressEvent(self, event):
            if event.button()==QtCore.Qt.MouseButton.LeftButton:
                pos = self.mapToView(event.pos())
                if pos is None: return
                mouse_real = np.array([pos.x(), pos.y()])/scale_factor
                dist = np.linalg.norm(mouse_real - head_position)
                if dist<0.2:
                    self.dragging=True
                event.accept()
            else:
                super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self.dragging:
                pos = self.mapToView(event.pos())
                if pos is None: return
                mouse_real = np.array([pos.x(), pos.y()])/scale_factor
                head_position[:] = mouse_real
                print("Dragging: calling update_plot()...")
                self.window().reload_filters()
                self.window().update_plot()
                pg.QtWidgets.QApplication.processEvents()
                event.accept()
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if event.button()==QtCore.Qt.MouseButton.LeftButton:
                self.dragging=False
                event.accept()
            else:
                super().mouseReleaseEvent(event)

    # ---------------------------------------------
    # Run
    # ---------------------------------------------
    list_audio_outputs()
    main_win = MainWindow()
    main_win.show()
    app.exec()

if __name__ == "__main__":
    setup_audio_routing()
    xtc_lab_processor()


