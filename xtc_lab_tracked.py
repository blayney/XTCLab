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

import numpy as np
import random
#from pythonosc.udp_client import SimpleUDPClient
import threading
import sounddevice as sd
import pyqtgraph as pg
import os
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QComboBox, QLabel, QPushButton
from PyQt6.QtCore import Qt
from scipy.fft import fft
from scipy.interpolate import interp1d
from math import sin, cos, radians
import xtc
import level_meter
from audio_engine import AudioEngine

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
    distance = 1.5
    left_az_deg = -30.0
    right_az_deg = 30.0

    head_position = np.array([0.0, 1.0])
    ideal_position = np.array([0.0, 1.0])
    ear_offset = 0.15
    scale_factor = 100.0
    c = 343.0  # Speed of sound

    xL = head_position[0] + distance * sin(radians(left_az_deg))
    yL = head_position[1] + distance * cos(radians(left_az_deg))
    xR = head_position[0] + distance * sin(radians(right_az_deg))
    yR = head_position[1] + distance * cos(radians(right_az_deg))

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
                    "measurement_channel": 0,
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

            self.fll = None
            self.flr = None
            self.frl = None
            self.frr = None
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
            # set minimum width of the plot
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

            self.ideal_marker = pg.ScatterPlotItem(
                pen=None, brush=pg.mkBrush(0, 255, 0, 150), size=15
            )
            ideal_x = ideal_position[0] * scale_factor
            ideal_y = ideal_position[1] * scale_factor
            self.ideal_marker.setData([ideal_x], [ideal_y])
            self.plot.addItem(self.ideal_marker)

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
            reset_button = QtWidgets.QPushButton("Reset Head Position")
            reset_button.clicked.connect(self.reset_head)
            center_layout.addWidget(reset_button)

            settings_button = QtWidgets.QPushButton("Settings")
            settings_button.clicked.connect(self.open_settings_menu)
            center_layout.addWidget(settings_button)
            inspect_hrir_button = QtWidgets.QPushButton("Inspect HRIRs")
            inspect_hrir_button.clicked.connect(self.open_hrir_inspection_modal)
            center_layout.addWidget(inspect_hrir_button)


            self.bypass_checkbox = QtWidgets.QCheckBox("Bypass Filters")
            self.bypass_checkbox.stateChanged.connect(self.update_audio_engine)
            center_layout.addWidget(self.bypass_checkbox)

            self.energy_checkbox = QtWidgets.QCheckBox("Show Energy Distribution")
            self.energy_checkbox.stateChanged.connect(self.toggle_energy_distribution)
            center_layout.addWidget(self.energy_checkbox)


            if self.config["binaural_device"] == "..." or self.config["playback_device"] == "...":
                QtCore.QTimer.singleShot(100, self.open_settings_menu)

            self.auto_range_topdown_plot()

            self.load_default_filters()

            self.update_plot()

        def update_audio_engine(self):
            if hasattr(self, "audio_engine"):
                del self.audio_engine
            with self.filter_lock:
                self.audio_engine = AudioEngine(
                    config=self.config,
                    meter_callback=self.update_meters,
                    f11=self.fll,
                    f12=self.flr,
                    f21=self.frl,
                    f22=self.frr,
                    bypass=self.bypass_checkbox.isChecked()
                )
            
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
            mic_combo = QComboBox()
            playback_left_combo = QComboBox()
            playback_right_combo = QComboBox()

            layout.addWidget(QLabel("Select Binaural Left Channel:"))
            layout.addWidget(binaural_left_combo)

            layout.addWidget(QLabel("Select Binaural Right Channel:"))
            layout.addWidget(binaural_right_combo)

            layout.addWidget(QLabel("Select Microphone Channel:"))
            layout.addWidget(mic_combo)

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
            measurement_device_combo.currentTextChanged.connect(lambda name: populate_channels(name, mic_combo, is_input=True))
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
                    "measurement_channel": mic_combo.currentData(),
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
            populate_channels(self.config["measurement_device"], mic_combo, is_input=True)
            populate_channels(self.config["playback_device"], [playback_left_combo, playback_right_combo], is_input=False)

            binaural_left_combo.setCurrentIndex(self.config["binaural_left_channel"] or 0)
            binaural_right_combo.setCurrentIndex(self.config["binaural_right_channel"] or 0)
            mic_combo.setCurrentIndex(self.config["measurement_channel"] or 0)
            playback_left_combo.setCurrentIndex(self.config["playback_left_channel"] or 0)
            playback_right_combo.setCurrentIndex(self.config["playback_right_channel"] or 0)
            hrtf_combo.setCurrentText(self.config["hrtf_file"])

            settings_dialog.exec()
            
        def update_meters(self, levels):
            for i, level in enumerate(levels[:len(self.meter_bars)]):
                self.meter_bars[i].set_level(level)

        def reload_filters(self):
            try:
                left_az_deg, right_az_deg = self.get_azimuth_pair()                
                HRIR_LL, HRIR_LR, sample_rate_l = xtc.extract_hrirs_sam(self.current_sofa_file, left_az_deg) # Left speaker left ear, Left speaker right ear
                HRIR_RL, HRIR_RR, sample_rate_r = xtc.extract_hrirs_sam(self.current_sofa_file, right_az_deg) # Right speaker left ear, Right speaker right ear
                self.hll = HRIR_LL
                self.hlr = HRIR_LR
                self.hrl = HRIR_RL
                self.hrr = HRIR_RR

                assert sample_rate_l == sample_rate_r, "Sample rates do not match!"
                self.samplerate = sample_rate_l[0]
                H_LL, H_LR, H_RL, H_RR = HRIR_LL, HRIR_LR, HRIR_RL, HRIR_RR
                self.fll, self.flr, self.frl, self.frr = xtc.generate_filter(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, 8192, samplerate=self.samplerate, debug=False)
                print("Filters reloaded successfully.")
            except Exception as e:
                print(f"Error loading default filters: {e}")
                
        def get_azimuth_pair(self):
            """
            Get the azimuth pair for the current head position.
            """
            for i, pos in enumerate(original_speaker_positions):
                angle = np.degrees(np.arctan2(pos[0] - head_position[0], pos[1] - head_position[1]))
                if i == 0:
                    source_azimuth_L = angle
                elif i == 1:
                    source_azimuth_R = angle
            source_azimuth_L
            source_azimuth_R
            print(f"Source azimuths: L={source_azimuth_L}, R={source_azimuth_R}")
            return source_azimuth_L, source_azimuth_R

                
        def load_default_filters(self):
            """
            Load the default P0275 HRIR filters.
            """
            print("Loading default HRTF.")
            self.current_sofa_file = "P0275_FreeFieldComp_48kHz.sofa"

            try:
                left_az_deg, right_az_deg = self.get_azimuth_pair()                
                HRIR_LL, HRIR_LR, sample_rate_l = xtc.extract_hrirs_sam(self.current_sofa_file, left_az_deg) # Left speaker left ear, Left speaker right ear
                HRIR_RL, HRIR_RR, sample_rate_r = xtc.extract_hrirs_sam(self.current_sofa_file, right_az_deg) # Right speaker left ear, Right speaker right ear
                self.hll = HRIR_LL
                self.hlr = HRIR_LR
                self.hrl = HRIR_RL
                self.hrr = HRIR_RR

                assert sample_rate_l == sample_rate_r, "Sample rates do not match!"
                self.samplerate = sample_rate_l[0]
                H_LL, H_LR, H_RL, H_RR = HRIR_LL, HRIR_LR, HRIR_RL, HRIR_RR
                self.fll, self.flr, self.frl, self.frr = xtc.generate_filter(H_LL, H_LR, H_RL, H_RR, original_speaker_positions, head_position, 8192, samplerate=self.samplerate, debug=False)
                print("Default filters loaded successfully.")
            except Exception as e:
                print(f"Error loading default filters: {e}")

        def toggle_energy_distribution(self):
            self.energy_distribution_enabled = self.energy_checkbox.isChecked()
            self.update_plot()

        def auto_range_topdown_plot(self):
            coords_x = list(original_speaker_positions[:,0]) + [head_position[0], ideal_position[0]]
            coords_y = list(original_speaker_positions[:,1]) + [head_position[1], ideal_position[1]]

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

        def compute_energy_distribution(self):
            """
            Fully simulate 2x2 MIMO crosstalk approach at each grid point
            in [-2..2] x [-2..2], sampling 100 points per axis.
            
            For each grid point:
            1) We do 'EarL=Impulse, EarR=0'  => measure net_right_ear power
            2) We do 'EarL=0,       EarR=Impulse' => measure net_left_ear  power
            We average those two contralateral powers, then store it as -10 log10(...) in a 2D array.

            Returns:
            xvals, yvals, energy_map
                where xvals, yvals are length=grid_size,
                and energy_map.shape = (grid_size, grid_size).
            """
            import numpy as np
            
            grid_size = 100
            xvals = np.linspace(-2, 2, grid_size)
            yvals = np.linspace(-2, 2, grid_size)
            energy_map = np.zeros((grid_size, grid_size), dtype=np.float32)

            ir_len = 512
            earL_imp = np.zeros(ir_len)
            earL_imp[0] = 1.0
            earR_imp = np.zeros(ir_len)
            earR_imp[0] = 1.0

            for j, yval in enumerate(yvals):
                for i, xval in enumerate(xvals):
                    test_pos = np.array([xval, yval])

                    # 1) 'EarL=Impulse'
                    netL_Limp, netR_Limp = self.compute_net_transfer_functions(test_pos,
                                                                            earL_imp,
                                                                            np.zeros_like(earL_imp))
                    # contralateral is netR_Limp => measure total power
                    contralateral_L_scenario = np.sum(netR_Limp**2)

                    # 2) 'EarR=Impulse'
                    netL_Rimp, netR_Rimp = self.compute_net_transfer_functions(test_pos,
                                                                            np.zeros_like(earR_imp),
                                                                            earR_imp)
                    # contralateral is netL_Rimp => measure total power
                    contralateral_R_scenario = np.sum(netL_Rimp**2)

                    # Average contralateral power
                    # (EarL=Imp => rightEar) + (EarR=Imp => leftEar)
                    mean_contralateral_power = 0.5 * (contralateral_L_scenario + contralateral_R_scenario)

                    # Convert power to dB, negative means good cancellation
                    # Also note that a single impulse has power=1 in time domain
                    # so do: energy_map[j,i] = -10 * log10(mean_contralateral + 1e-12), etc.
                    energy_map[j, i] = -10.0 * np.log10(mean_contralateral_power + 1e-12)

            return xvals, yvals, energy_map
        
        def get_acoustics_data(self):
            """
            Our broad goal here is to show the following transfer functions:
            1) Left System Input to Left Ear (Left Ipsilateral) (left Graph)
            2) Right System Input to Left Ear (Left Contralateral) (Left Graph)
            3) Right System Input to Right Ear (Right Ipsilateral) (Right Graph)
            4) Left System Input to Right Ear (Right Contralateral) (Right Graph)

            To do this, we simulate two impulses, one corresponding to each system input.
            We then apply the following "sub" transfer functions to each of these.

            This system of equations looks like E = HPFS
            where E is the ear pressure vector
            H is the HRTF matrix
            P is the delay matrix pulled out of the HRTF
            F is the filter matrix
            S is the system input vector

            We can calculate the ear pressure:
                Starting with an impulse for each system input, we then:
                1) Convolve this with the filter matrix to get FS
                2) Convolve this matrix with the delay matrix to get the pre-hrtf signal at the head
                3) convolve this with the HRTF matrix to get the ear pressure vectors
            
            The result is four transfer functions, which we can plot.
            This is in the format:
                (Left_Ear_Left_Channel, Left_Ear_Right_Channel),
                (Right_Ear_Right_Channel, Right_Ear_Left_Channel),
                (t, f11_centered),
                (t, f12_centered),
                (t, f21_centered),
                (t, f22_centered)
            """
            def zero_shift(x, d):
                if d >= 0:
                    return np.concatenate([np.zeros(d), x])[:len(x)]
                else:
                    return x[-d:]
                
            delay_samples_l = int(np.round(
                (np.linalg.norm(original_speaker_positions[0] - head_position) / c) * self.samplerate
            ))
            delay_samples_r = int(np.round(
                (np.linalg.norm(original_speaker_positions[1] - head_position) / c) * self.samplerate
            ))

            with self.filter_lock:
                # Compute propagation delays to the head (in samples)
                N = (
                    max(len(self.fll), len(self.flr), len(self.frl), len(self.frr))
                    + max(len(self.hll), len(self.hlr), len(self.hrl), len(self.hrr))
                )

                # We could simulate the impulse, but we don't need to
                # Because it's the same as multiplying by 1.

                H_ll = np.fft.fft(self.hll, N)
                H_lr = np.fft.fft(self.hlr, N)
                H_rl = np.fft.fft(self.hrl, N)
                H_rr = np.fft.fft(self.hrr, N)

                F_ll = np.fft.fft(self.fll, N)
                F_lr = np.fft.fft(self.flr, N)
                F_rl = np.fft.fft(self.frl, N)
                F_rr = np.fft.fft(self.frr, N)

                freqs = np.fft.fftfreq(N, d=1/self.samplerate)
                omega = 2 * np.pi * freqs

                exp_L = np.exp(1j * omega * -delay_samples_l/self.samplerate)    # for the left speaker
                exp_R = np.exp(1j * omega * -delay_samples_r/self.samplerate)    # for the right speaker

                X_LL = H_ll*F_ll + H_lr*F_rl
                X_LR = H_ll*F_lr + H_lr*F_rr
                X_RL = H_rl*F_ll + H_rr*F_rl
                X_RR = H_rl*F_lr + H_rr*F_rr

                x_LL = np.fft.ifft(X_LL)
                x_LR = np.fft.ifft(X_LR)
                x_RR = np.fft.ifft(X_RR)
                x_RL = np.fft.ifft(X_RL)

                Left_Ear_Left_Channel = x_LL
                Left_Ear_Right_Channel = x_LR
                Right_Ear_Right_Channel = x_RR
                Right_Ear_Left_Channel = x_RL

                # Build unit impulses long enough to carry filter + HRIR
                # N_imp = (
                #     max(len(self.fll), len(self.flr), len(self.frl), len(self.frr))
                #     + max(len(self.hll), len(self.hlr), len(self.hrl), len(self.hrr))
                # )
                # S_L = np.zeros(N_imp); S_L[0] = 1.0
                # S_R = np.zeros(N_imp); S_R[0] = 1.0

                # # --- Response at ears for left‐channel impulse (ipsilateral = flat) ---
                # # Left input → Left speaker, then propagate to head
                # X_L_from_L = zero_shift(
                #     np.convolve(S_L, self.fll, mode='full'),
                #     delay_samples_l
                # )
                # print("  X_L_from_L first non-zero index:", np.argmax(np.abs(X_L_from_L)))
                # print("delay L, ", delay_samples_l)

                # # Left input → Right speaker, then propagate to head
                # X_R_from_L = zero_shift(
                #     np.convolve(S_L, self.frl, mode='full'),
                #     delay_samples_r
                # )
                # # Left‐ear response (ipsilateral)
                # E_Lipsi = (
                #     np.convolve(X_L_from_L, self.hll, mode='full')
                #     + np.convolve(X_R_from_L, self.hrl, mode='full')
                # )
                # # Right‐ear response to left input (contralateral for right ear)
                # E_Rcontra = (
                #     np.convolve(X_L_from_L, self.hlr, mode='full')
                #     + np.convolve(X_R_from_L, self.hrr, mode='full')
                # )

                # # --- Response at ears for right‐channel impulse (contralateral = cancelled) ---
                # X_L_from_R = zero_shift(
                #     np.convolve(S_R, self.flr, mode='full'),
                #     delay_samples_l
                # )
                # X_R_from_R = zero_shift(
                #     np.convolve(S_R, self.frr, mode='full'),
                #     delay_samples_r
                # )
                # # Left‐ear response to right input (contralateral)
                # E_Lcontra = (
                #     np.convolve(X_L_from_R, self.hll, mode='full')
                #     + np.convolve(X_R_from_R, self.hrl, mode='full')
                # )
                # # Right‐ear response (ipsilateral)
                # E_Ripsi = (
                #     np.convolve(X_L_from_R, self.hlr, mode='full')
                #     + np.convolve(X_R_from_R, self.hrr, mode='full')
                # )

                # # Assign into the four TF outputs for plotting
                # Left_Ear_Left_Channel  = E_Lipsi
                # Left_Ear_Right_Channel = E_Lcontra
                # Right_Ear_Right_Channel = E_Ripsi
                # Right_Ear_Left_Channel  = E_Rcontra

                if self.fll is not None:
                    def align_by_peak(ir):
                        peak_index = np.argmax(np.abs(ir))
                        centered = np.roll(ir, len(ir)//2 - peak_index)
                        return centered

                    f11_centered = align_by_peak(self.fll)
                    f12_centered = align_by_peak(self.flr)
                    f21_centered = align_by_peak(self.frl)
                    f22_centered = align_by_peak(self.frr)

                    t = np.arange(len(f11_centered)) / self.samplerate

                    self.tf_curves[2][0].setData(t, f11_centered)  # f11
                    self.tf_curves[3][0].setData(t, f12_centered)  # f12
                    self.tf_curves[4][0].setData(t, f21_centered)  # f21
                    self.tf_curves[5][0].setData(t, f22_centered)  # f22

            # with self.filter_lock:

            #     # If the filters haven't been computed yet, return empty arrays
            #     if any(f is None for f in [self.f11_time, self.f12_time, self.f21_time, self.f22_time]):
            #         return [(np.array([0.0]), np.array([0.0]))] * 4
                
            #     # Step 1: Start with "inputs" to the system
            #     Left_Input = np.zeros(4096)
            #     Right_Input = np.zeros(4096)
            #     Left_Input[0] = 1.0
            #     Right_Input[0] = 1.0

            #     # Step 2: Convolve with the filters
            #     Left_Speaker_Left_Channel = np.convolve(Left_Input, self.f11_time, mode='full')
            #     Right_Speaker_Left_Channel = np.convolve(Left_Input, self.f21_time, mode='full')
            #     Left_Speaker_Right_Channel = np.convolve(Right_Input, self.f12_time, mode='full')
            #     Right_Speaker_Right_Channel = np.convolve(Right_Input, self.f22_time, mode='full')

            #     # Step 3: Delay the signals to simulate travel to the head, using the distance between the head and the speakers
            #     delay_samples_l = int(np.round((np.linalg.norm(original_speaker_positions[0] - head_position) / c) * self.samplerate))
            #     delay_samples_r = int(np.round((np.linalg.norm(original_speaker_positions[1] - head_position) / c) * self.samplerate))
            #     Left_Speaker_Left_Channel = np.roll(Left_Speaker_Left_Channel, delay_samples_l)
            #     Right_Speaker_Left_Channel = np.roll(Right_Speaker_Left_Channel, delay_samples_r)
            #     Left_Speaker_Right_Channel = np.roll(Left_Speaker_Right_Channel, delay_samples_l)
            #     Right_Speaker_Right_Channel = np.roll(Right_Speaker_Right_Channel, delay_samples_r)

            #     # Step 4: Compute left ear transfer functions
            #     Left_Ear_Left_Channel_Left_Speaker = np.convolve(Left_Speaker_Left_Channel, self.h_ll, mode='full')
            #     Left_Ear_Left_Channel_Right_Speaker = np.convolve(Right_Speaker_Left_Channel, self.h_rl, mode='full')
            #     # this is the left ear ipsilateral:
            #     Left_Ear_Left_Channel = Left_Ear_Left_Channel_Left_Speaker + Left_Ear_Left_Channel_Right_Speaker
            #     Left_Ear_Right_Channel_Left_Speaker = np.convolve(Left_Speaker_Right_Channel, self.h_ll, mode='full')
            #     Left_Ear_Right_Channel_Right_Speaker = np.convolve(Right_Speaker_Right_Channel, self.h_rl, mode='full')
            #     # this is the left ear contralateral:
            #     Left_Ear_Right_Channel = Left_Ear_Right_Channel_Left_Speaker + Left_Ear_Right_Channel_Right_Speaker

            #     # Step 5: Compute right ear transfer functions
            #     Right_Ear_Left_Channel_Left_Speaker = np.convolve(Left_Speaker_Left_Channel, self.h_lr, mode='full')
            #     Right_Ear_Left_Channel_Right_Speaker = np.convolve(Right_Speaker_Left_Channel, self.h_rr, mode='full')
            #     # this is the right ear contralateral:
            #     Right_Ear_Left_Channel = Right_Ear_Left_Channel_Left_Speaker + Right_Ear_Left_Channel_Right_Speaker
            #     Right_Ear_Right_Channel_Left_Speaker = np.convolve(Left_Speaker_Right_Channel, self.h_lr, mode='full')
            #     Right_Ear_Right_Channel_Right_Speaker = np.convolve(Right_Speaker_Right_Channel, self.h_rr, mode='full')
            #     # this is the right ear ipsilateral:
            #     Right_Ear_Right_Channel = Right_Ear_Right_Channel_Left_Speaker + Right_Ear_Right_Channel_Right_Speaker

                # Plot 4 IRs: f11, f12, f21, f22
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

            import matplotlib.pyplot as plt

            def zero_shift(x,d):
                if d>=0: return np.concatenate([np.zeros(d), x])[:len(x)]
                return x[-d:]
            def align_to_peak(ir):
                p = np.argmax(np.abs(ir)); return np.roll(ir, -p)

            # # you already have H_ll, H_lr, … and FLL, FLR, etc.
            # freqs = np.fft.fftfreq(N,1/self.samplerate)[:N//2]
            # # 1) HF before any delay embed
            # H_freq = np.array([[H_ll, H_lr],[H_rl, H_rr]])      # shape (2,2,N)
            # F_raw  = np.array([[FLL, FLR],[FRL, FRR_raw]])
            # HF_raw = np.einsum('imk,mjk->ijk', H_freq, F_raw)
            # # 2) HF after you multiplied in your exp_L/exp_R
            # F_emb  = np.array([[FLL, FLR],[FRL, FRR]])
            # HF_del = np.einsum('imk,mjk->ijk', H_freq, F_emb)

            # fig, axes = plt.subplots(2,2, figsize=(8,6))
            # for (ax, data, title) in zip(axes.flat,
            #                             [HF_raw[0,1,:], HF_del[0,1,:], HF_raw[1,0,:], HF_del[1,0,:]],
            #                             ['Raw HF₁₂','Delayed HF₁₂','Raw HF₂₁','Delayed HF₂₁']):
            #     mag = 20*np.log10(np.clip(np.abs(data[:N//2]),1e-16,None))
            #     ax.plot(freqs, mag)
            #     ax.set_title(title); ax.set_ylim(-120,5); ax.set_xlim(0,20000)
            # plt.tight_layout(); plt.show()            plt.legend(); plt.title("Debug: aligned+windowed TFs"); plt.show()
            
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
                self.energy_item.setImage(energy_map.T)
                self.energy_item.setRect(pg.QtCore.QRectF(x_min, y_min, width, height))
                self.energy_item.setVisible(True)
            else:
                self.energy_item.setVisible(False)

            # Head / Ears
            hx = head_position[0]*scale_factor
            hy = head_position[1]*scale_factor
            self.head_circle.setData([hx],[hy])

            L_ear = head_position + np.array([-ear_offset/2, 0.0])
            R_ear = head_position + np.array([ ear_offset/2, 0.0])
            self.ear_dots.setData(
                [L_ear[0]*scale_factor, R_ear[0]*scale_factor],
                [L_ear[1]*scale_factor, R_ear[1]*scale_factor]
            )

            # Lines
            connections = [
                (original_speaker_positions[0], L_ear),
                (original_speaker_positions[0], R_ear),
                (original_speaker_positions[1], L_ear),
                (original_speaker_positions[1], R_ear),
            ]

            # Update angle labels
            for i, pos in enumerate(original_speaker_positions):
                angle = np.degrees(np.arctan2(pos[0] - head_position[0], pos[1] - head_position[1]))
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


        def compute_net_transfer_functions(self, head_pos, earL, earR):
            """
            Same logic as apply_xtc_filters_mimo, but uses self.f11_time, etc.
            """
            # Convolve ear signals with the filters
            with self.filter_lock:
                if any(f is None for f in [self.fll, self.flr, self.frl, self.frr]):
                    return np.zeros_like(earL), np.zeros_like(earR)
                spkL_from_left = np.convolve(earL, self.fll, mode='same')
                spkL_from_right= np.convolve(earR, self.flr, mode='same')
                speaker_L = spkL_from_left + spkL_from_right

                spkR_from_left = np.convolve(earL, self.frl, mode='same')
                spkR_from_right= np.convolve(earR, self.frr, mode='same')
                speaker_R = spkR_from_left + spkR_from_right

                # Delays
                L_ear = head_pos + np.array([-ear_offset/2, 0.0])
                R_ear = head_pos + np.array([ ear_offset/2, 0.0])
                distL_ear = np.linalg.norm(original_speaker_positions - L_ear, axis=1)
                distR_ear = np.linalg.norm(original_speaker_positions - R_ear, axis=1)

                delayL_samps = (distL_ear / c * self.samplerate).astype(int)
                delayR_samps = (distR_ear / c * self.samplerate).astype(int)

                spk0_left = np.roll(speaker_L, delayL_samps[0])
                spk1_left = np.roll(speaker_R, delayL_samps[1])
                net_left = spk0_left + spk1_left

                spk0_right= np.roll(speaker_L, delayR_samps[0])
                spk1_right= np.roll(speaker_R, delayR_samps[1])
                net_right= spk0_right + spk1_right
            return net_left, net_right

        def reset_head(self):
            head_position[:] = ideal_position
            self.update_plot()
            
        def regenerate_filters_from_current_head_position(self):
            self.reload_filters()

        def open_hrir_inspection_modal(self):
            try:
                if self.hll is not None:
                    data_4 = {
                        "H_LL": self.hll,
                        "H_LR": self.hlr,
                        "H_RL": self.hrl,
                        "H_RR": self.hrr,
                        "samplerate": self.samplerate
                    }
                else:
                    data_4 = xtc.extract_hrirs_sam(self.current_sofa_file, left_az=-30.0, right_az=30.0)
                    print("Opened modal and extracted HRIRs")
                H_LL, H_LR, H_RL, H_RR = data_4["H_LL"], data_4["H_LR"], data_4["H_RL"], data_4["H_RR"]
                samplerate = data_4["samplerate"]
                with self.filter_lock:
                                    f11, f12, f21, f22 = xtc.generate_filter(
                                        H_LL, H_LR, H_RL, H_RR,
                                        head_position=head_position,
                                        speaker_positions=original_speaker_positions,
                                        filter_length=8192,
                                        samplerate=samplerate,
                                    )
            except Exception as e:
                print(f"Error loading HRIRs: {e}")
                return

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("HRIRs and Inverted Filters")
            layout = QtWidgets.QVBoxLayout(dialog)

            plot_widget = pg.GraphicsLayoutWidget()
            layout.addWidget(plot_widget)

            def plot_ir_row(title, left_data, right_data):
                p = plot_widget.addPlot(title=title)
                t = np.arange(len(left_data)) / samplerate
                p.plot(t, left_data, pen='r', name="Left")
                p.plot(t, right_data, pen='b', name="Right")
                p.showGrid(x=True, y=True)
                plot_widget.nextRow()

            plot_ir_row("HRIRs: H_LL and H_RR", H_LL, H_RR)
            plot_ir_row("HRIRs: H_LR and H_RL", H_LR, H_RL)
            plot_ir_row("Inverted Filters: f11 and f22", f11, f22)
            plot_ir_row("Inverted Filters: f12 and f21", f12, f21)

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

