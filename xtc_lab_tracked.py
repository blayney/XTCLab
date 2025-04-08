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

            self.setWindowTitle("XTC Lab Processor - Binaural Audio Processing")
            self.resize(1920, 1080)

            # Instance variables
            self.f11_time = None
            self.f12_time = None
            self.f21_time = None
            self.f22_time = None
            self.current_sofa_file = "KEMAR_HRTF_FFComp.sofa" 
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
            channel_labels = ["IL", "IR", "Mic", "OL", "OR"]
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
                if label == "Mic" or label == "IR":
                    separator = QtWidgets.QFrame()
                    separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
                    separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
                    mixer_layout.addWidget(separator)

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
                "Left Input → Ears (Transfer Function)",
                "Right Input → Ears (Transfer Function)",
                "Left Input Impulse Responses (f11, f12)",
                "Right Input Impulse Responses (f21, f22)"
            ]

            colors = ['r', 'b']
            for i, title in enumerate(titles):
                pw = pg.PlotWidget(title=title)
                pw.setLabel("bottom", "Frequency (Hz)" if i < 2 else "Time (s)")
                pw.setLabel("left", "Magnitude (dB)" if i < 2 else "Amplitude")
                if i < 2:
                    pw.enableAutoRange(axis='y', enable=True)
                curve1 = pw.plot(pen=colors[0], name="Left Ear")
                curve2 = pw.plot(pen=colors[1], name="Right Ear")
                pw.addLegend()
                self.tf_plots.append(pw)
                self.tf_curves.append((curve1, curve2))
                fft_layout.addWidget(pw, i // 2, i % 2)
            self.plot = DraggablePlot(title="Geometric View")
            self.plot.setAspectLocked(True)

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

                txt = pg.TextItem(anchor=(0.5, 1.0))
                txt.setColor("gray")
                self.speaker_ear_labels.append(txt)
                self.plot.addItem(txt)


            # Layout
            center_widget = QtWidgets.QWidget()
            center_layout = QtWidgets.QVBoxLayout(center_widget)
            self.setCentralWidget(center_widget)
            center_layout.addWidget(self.plot)
            self.edit_reg_button = QtWidgets.QPushButton("Edit Regularization Profile")
            self.edit_reg_button.clicked.connect(self.open_regularization_editor)
            center_layout.addWidget(self.edit_reg_button)
            reset_button = QtWidgets.QPushButton("Reset Head Position")
            reset_button.clicked.connect(self.reset_head)
            center_layout.addWidget(reset_button)

            settings_button = QtWidgets.QPushButton("Settings")
            settings_button.clicked.connect(self.open_settings_menu)
            center_layout.addWidget(settings_button)
            inspect_hrir_button = QtWidgets.QPushButton("Inspect HRIRs")
            inspect_hrir_button.clicked.connect(self.open_hrir_inspection_modal)
            center_layout.addWidget(inspect_hrir_button)

            gen_button = QtWidgets.QPushButton("Generate HRIR & Reload Filters")
            gen_button.clicked.connect(self.regenerate_filters_from_current_head_position)            
            center_layout.addWidget(gen_button)

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
                    f11=self.f11_time,
                    f12=self.f12_time,
                    f21=self.f21_time,
                    f22=self.f22_time,
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
                self.reload_filters_with_current_regularization()
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

        def update_regularization(self):
                """
                Update the regularization parameter from the slider and reload filters.
                """
                slider_value = self.reg_slider.value()
                self.regularization = 10 ** (slider_value / 100 - 2)  # Convert to log scale
                self.reg_label.setText(f"Regularization: {self.regularization:.4f}")

                # Reload filters with the updated regularization value
                with self.filter_lock:
                    self.reload_filters_with_current_regularization()
                    
        def reload_filters_with_current_regularization(self):
            print("Generating filters using SOFA HRIRs with automated frequency-dependent regularization...")

            try:
                data_4 = xtc.extract_4_ir_sofa(self.current_sofa_file, left_az=-30.0, right_az=30.0)
                samplerate = data_4["samplerate"]
                H_LL, H_LR, H_RL, H_RR = xtc.align_impulse_responses(
                    data_4["H_LL"], data_4["H_LR"], data_4["H_RL"], data_4["H_RR"]
                )

                c = 343.0
                L_ear = head_position + np.array([-ear_offset / 2, 0.0])
                R_ear = head_position + np.array([ ear_offset / 2, 0.0])

                def get_itd(dist):
                    return int(round(float(np.asarray(dist / c * samplerate).item())))

                itd_LL = get_itd(np.linalg.norm(original_speaker_positions[0] - L_ear))
                itd_LR = get_itd(np.linalg.norm(original_speaker_positions[0] - R_ear))
                itd_RL = get_itd(np.linalg.norm(original_speaker_positions[1] - L_ear))
                itd_RR = get_itd(np.linalg.norm(original_speaker_positions[1] - R_ear))

                max_itd = max(itd_LL, itd_LR, itd_RL, itd_RR)

                def delay_pad(ir, delay_samples):
                    padded = np.zeros(max_itd + len(ir))
                    padded[delay_samples:delay_samples + len(ir)] = ir
                    return padded

                H_LL = delay_pad(H_LL, itd_LL)
                H_LR = delay_pad(H_LR, itd_LR)
                H_RL = delay_pad(H_RL, itd_RL)
                H_RR = delay_pad(H_RR, itd_RR)

                self.samplerate = samplerate

                gamma_db = 7.0  # User-adjustable target in dB
                gamma = 10**(gamma_db / 20.0)

                def regularization(freqs):
                    omega = 2 * np.pi * freqs
                    g = 1.0  # Assuming equal amplitude
                    tau_c = ear_offset / c
                    epsilon = 1e-10

                    cos_term = np.cos(omega * tau_c)
                    sqrt_I = np.sqrt(g**2 - 2*g*cos_term + 1)
                    sqrt_II = np.sqrt(g**2 + 2*g*cos_term + 1)

                    beta_I = -g**2 + 2*g*cos_term + (sqrt_I / gamma) - 1
                    beta_II = -g**2 - 2*g*cos_term + (sqrt_II / gamma) - 1

                    S_o = 1 / sqrt_I
                    S_i = 1 / sqrt_II
                    S_p = np.maximum(S_o, S_i)

                    beta = np.where(S_p < gamma, 0.0, np.where(S_o >= S_i, beta_I, beta_II))
                    beta = np.maximum(beta, epsilon)

                    return beta.astype(np.float64)

                with self.filter_lock:
                    self.f11_time, self.f12_time, self.f21_time, self.f22_time = xtc.generate_xtc_filters_mimo(
                        H_LL, H_LR, H_RL, H_RR,
                        samplerate=self.samplerate,
                        regularization=regularization
                    )
                    self.update_plot()
            except Exception as e:
                print(f"Error generating filters with frequency-dependent regularization: {e}")

                
        def regenerate_filters_from_current_head_position(self):
            """
            Regenerate filters based on the current head position and reload them.
            """
            print("Regenerating filters from current head position...")
            self.reload_filters_with_current_regularization()      
        def load_default_filters(self):
            """
            Load the default KEMAR HRIR filters.
            """
            print("Loading default KEMAR filters.")
            self.current_sofa_file = "KEMAR_HRTF_FFComp.sofa"

            try:
                data_4 = xtc.extract_4_ir_sofa(self.current_sofa_file, left_az=-30.0, right_az=30.0)
                H_LL, H_LR, H_RL, H_RR = xtc.align_impulse_responses(
                    data_4["H_LL"], data_4["H_LR"], data_4["H_RL"], data_4["H_RR"]
                )
                self.samplerate = data_4["samplerate"]

                self.f11_time, self.f12_time, self.f21_time, self.f22_time = xtc.generate_xtc_filters_mimo(
                    H_LL, H_LR, H_RL, H_RR,
                    samplerate=self.samplerate,
                regularization=lambda freqs: np.full_like(freqs, self.regularization)
                )
                print("Default filters loaded successfully.")
            except Exception as e:
                print(f"Error loading default filters: {e}")
                
        def open_regularization_editor(self):
            import matplotlib.pyplot as plt
            from scipy.interpolate import interp1d

            # Sample frequencies (log-spaced from 20 Hz to 20 kHz), only if not already set
            if not hasattr(self, "reg_freqs") or not hasattr(self, "reg_values"):
                self.reg_freqs = np.logspace(np.log10(20), np.log10(20000), 10)
                self.reg_values = np.ones_like(self.reg_freqs) * self.regularization

            fig, ax = plt.subplots()
            line, = ax.plot(self.reg_freqs, self.reg_values, marker='o')
            ax.set_xscale("log")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Regularization")
            ax.set_title("Frequency-Dependent Regularization")
            ax.grid(True)

            from threading import Timer
            
            debounce_timer = [None]
            
            def schedule_filter_update():
                if debounce_timer[0] is not None:
                    debounce_timer[0].cancel()
                debounce_timer[0] = Timer(0.2, finalize_update)
                debounce_timer[0].start()
            
            def finalize_update():
                self.regularization_interp = interp1d(
                    self.reg_freqs, self.reg_values, bounds_error=False, fill_value="extrapolate"
                )
                self.reload_filters_with_current_regularization()
                import json
                with open("regularization_profile.json", "w") as f:
                    json.dump({
                        "freqs": self.reg_freqs.tolist(),
                        "values": self.reg_values.tolist()
                    }, f)
            
            def on_click(event):
                if event.inaxes != ax:
                    return
                closest = np.argmin(np.abs(self.reg_freqs - event.xdata))
                self.reg_values[closest] = event.ydata
                line.set_ydata(self.reg_values)
                fig.canvas.draw()
                schedule_filter_update()

            fig.canvas.mpl_connect("button_press_event", on_click)

            def on_close(event):
                finalize_update()
            fig.canvas.mpl_connect("close_event", on_close)
            plt.show()

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
                    netL_Limp, netR_Limp = self.apply_xtc_filters_mimo_local(test_pos,
                                                                            earL_imp,
                                                                            np.zeros_like(earL_imp))
                    # contralateral is netR_Limp => measure total power
                    contralateral_L_scenario = np.sum(netR_Limp**2)

                    # 2) 'EarR=Impulse'
                    netL_Rimp, netR_Rimp = self.apply_xtc_filters_mimo_local(test_pos,
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
                self.speaker_ear_labels[i].setPos(mid_scaled[0], mid_scaled[1])
                self.speaker_ear_labels[i].setText(f"{dist_m:.2f} m\n{delay_ms:.2f} ms")

            # Recompute the 4 lines for crosstalk using the MIMO filters: self.fXX_time
            lines = self.compute_transfer_functions_mimo_local()
            for i, (freqs, mags) in enumerate(lines[:2]):
                if isinstance(mags, tuple) and len(mags) == 2:
                    self.tf_curves[i][0].setData(np.asarray(freqs), np.asarray(mags[0]))
                    self.tf_curves[i][1].setData(np.asarray(freqs), np.asarray(mags[1]))

        def compute_transfer_functions_mimo_local(self):
            """
             Compute transfer functions for the current MIMO filters (self.f11_time, etc.)
             and return the frequency response (magnitude in dB) for each path.
             """
            if any(f is None for f in [self.f11_time, self.f12_time, self.f21_time, self.f22_time]):
                return [(np.array([0.0]), np.array([0.0]))] * 4
            sig_len = 1024  # Length of the impulse response
            earL_imp = np.zeros(sig_len)
            earL_imp[0] = 1.0  # Impulse for left ear
            earR_imp = np.zeros(sig_len)
            earR_imp[0] = 1.0  # Impulse for right ear

            # Apply XTC filters
            netL_Limp, netR_Limp = self.apply_xtc_filters_mimo_local(head_position, earL_imp, np.zeros_like(earL_imp))
            netL_Rimp, netR_Rimp = self.apply_xtc_filters_mimo_local(head_position, np.zeros_like(earR_imp), earR_imp)
            # FFT for magnitude and frequency axis
            def fft_mag_db(signal):
                fft_len = len(signal)
                freq_axis = np.fft.fftfreq(fft_len, d=1.0 / self.samplerate)[:fft_len // 2]
                magnitude = np.abs(fft(signal))[:fft_len // 2]
                magnitude_db = 20 * np.log10(magnitude + 1e-12)
                return freq_axis, magnitude_db

            # Compute transfer functions
            f_LL, m_LL = fft_mag_db(netL_Limp)
            f_LR, m_LR = fft_mag_db(netR_Limp)
            f_RL, m_RL = fft_mag_db(netL_Rimp)
            f_RR, m_RR = fft_mag_db(netR_Rimp)

            self.tf_curves[0][0].setData(f_LL, m_LL)
            self.tf_curves[0][1].setData(f_LR, m_LR)
            self.tf_curves[1][0].setData(f_RL, m_RL)
            self.tf_curves[1][1].setData(f_RR, m_RR)

            # Plot IRs
            if self.f11_time is not None:
                def align_by_peak(ir):
                    peak_index = np.argmax(np.abs(ir))
                    centered = np.roll(ir, len(ir)//2 - peak_index)
                    return centered

                f11_centered = align_by_peak(self.f11_time)
                f12_centered = align_by_peak(self.f12_time)
                f21_centered = align_by_peak(self.f21_time)
                f22_centered = align_by_peak(self.f22_time)

                t = np.arange(len(f11_centered)) / self.samplerate
                self.tf_curves[2][0].setData(t, f11_centered)
                self.tf_curves[2][1].setData(t, f12_centered)
                self.tf_curves[3][0].setData(t, f21_centered)
                self.tf_curves[3][1].setData(t, f22_centered)

            return [(f_LL, m_LL), (f_LR, m_LR), (f_RL, m_RL), (f_RR, m_RR)]

        def apply_xtc_filters_mimo_local(self, head_pos, earL, earR):
            """
            Same logic as apply_xtc_filters_mimo, but uses self.f11_time, etc.
            """
            # Convolve ear signals with the filters
            with self.filter_lock:
                if any(f is None for f in [self.f11_time, self.f12_time, self.f21_time, self.f22_time]):
                    return np.zeros_like(earL), np.zeros_like(earR)
                spkL_from_left = np.convolve(earL, self.f11_time, mode='same')
                spkL_from_right= np.convolve(earR, self.f12_time, mode='same')
                speaker_L = spkL_from_left + spkL_from_right

                spkR_from_left = np.convolve(earL, self.f21_time, mode='same')
                spkR_from_right= np.convolve(earR, self.f22_time, mode='same')
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
            self.reload_filters_with_current_regularization()

        def open_hrir_inspection_modal(self):
            try:
                data_4 = xtc.extract_4_ir_sofa(self.current_sofa_file, left_az=-30.0, right_az=30.0)
                print("Opened modal and extracted HRIRs")
                H_LL, H_LR, H_RL, H_RR = xtc.align_impulse_responses(
                    data_4["H_LL"], data_4["H_LR"], data_4["H_RL"], data_4["H_RR"]
                )
                print("Aligned HRIRs")
                samplerate = data_4["samplerate"]
                f11, f12, f21, f22 = xtc.generate_xtc_filters_mimo(
                    H_LL, H_LR, H_RL, H_RR,
                    samplerate=samplerate,
                    regularization=lambda freqs: np.full_like(freqs, self.regularization)
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

