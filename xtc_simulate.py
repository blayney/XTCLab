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
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget
from PyQt6.QtCore import Qt
from scipy.fft import fft
from math import sin, cos, radians
import xtc

def simulate_xtc():
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
    scale_factor = 10.0
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

            self.samplerate = None

            self.setWindowTitle("XTC 2×2 MIMO Simulation + HRIR Generation")
            self.resize(1200, 800)

            # Instance variables
            self.f11_time = None
            self.f12_time = None
            self.f21_time = None
            self.f22_time = None
            self.current_sofa_file = "KEMAR_HRTF_FFComp.sofa" 
            self.regularization = 0.01

            self.energy_distribution_enabled = False

            # FFT Dock
            self.fftDock = QDockWidget("Crosstalk FFTs", self)
            features = (QDockWidget.DockWidgetFeature.DockWidgetMovable |
                        QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.fftDock.setFeatures(features)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.fftDock)

            fft_container = QtWidgets.QWidget()
            fft_layout = QtWidgets.QGridLayout(fft_container)
            self.fftDock.setWidget(fft_container)

            self.tf_plots = []
            self.tf_curves = []
            plot_titles = [
                "SpeakerL → netLeftEar",
                "SpeakerL → netRightEar",
                "SpeakerR → netLeftEar",
                "SpeakerR → netRightEar"
            ]
            for i, t_str in enumerate(plot_titles):
                pw = pg.PlotWidget(title=t_str)
                pw.setLabel("bottom", "Frequency (Hz)")
                pw.setLabel("left", "Magnitude (dB)")
                pw.setYRange(-60, 10)
                curve = pw.plot(pen="b")
                self.tf_plots.append(pw)
                self.tf_curves.append(curve)
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

            self.regularization = 0.01  # Default 
            self.reg_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            self.reg_slider.setMinimum(0)
            self.reg_slider.setMaximum(200)
            self.reg_slider.setValue(100)
            self.reg_slider.valueChanged.connect(self.update_regularization)
            self.reg_label = QtWidgets.QLabel(f"Regularization: {self.regularization:.4f}")
            reg_layout = QtWidgets.QHBoxLayout()
            reg_layout.addWidget(QtWidgets.QLabel("Regularization:"))
            reg_layout.addWidget(self.reg_slider)
            reg_layout.addWidget(self.reg_label)

            # Layout
            center_widget = QtWidgets.QWidget()
            center_layout = QtWidgets.QVBoxLayout(center_widget)
            self.setCentralWidget(center_widget)
            center_layout.addWidget(self.plot)
            center_layout.addLayout(reg_layout)
            reset_button = QtWidgets.QPushButton("Reset Head Position")
            reset_button.clicked.connect(self.reset_head)
            center_layout.addWidget(reset_button)

            gen_button = QtWidgets.QPushButton("Generate HRIR & Reload Filters")
            gen_button.clicked.connect(self.generate_hrir_and_reload_filters)
            center_layout.addWidget(gen_button)

            self.energy_checkbox = QtWidgets.QCheckBox("Show Energy Distribution")
            self.energy_checkbox.stateChanged.connect(self.toggle_energy_distribution)
            center_layout.addWidget(self.energy_checkbox)

            self.auto_range_topdown_plot()

            self.load_default_filters()

            self.update_plot()

        def update_regularization(self):
            """
            Update the regularization parameter from the slider and reload filters.
            """
            slider_value = self.reg_slider.value()
            self.regularization = 10 ** (slider_value / 100 - 2)  # Convert to log scale
            self.reg_label.setText(f"Regularization: {self.regularization:.4f}")

            # Reload filters with the updated regularization value
            self.reload_filters_with_current_regularization()

        def reload_filters_with_current_regularization(self):
            """
            Reload the filters using the current SOFA file and updated regularization value.
            """
            print(f"Reloading filters from {self.current_sofa_file} with regularization: {self.regularization:.4f}")

            try:
                data_4 = xtc.extract_4_ir_sofa(self.current_sofa_file, left_az=-30.0, right_az=30.0)
                H_LL = data_4["H_LL"]
                H_LR = data_4["H_LR"]
                H_RL = data_4["H_RL"]
                H_RR = data_4["H_RR"]

                self.f11_time, self.f12_time, self.f21_time, self.f22_time = xtc.generate_xtc_filters_mimo(
                    H_LL, H_LR, H_RL, H_RR,
                    samplerate=self.samplerate,
                    regularization=self.regularization
                )
                print("Filters reloaded successfully.")
                self.update_plot()
            except Exception as e:
                print(f"Error reloading filters: {e}")

        def load_default_filters(self):
            """
            Load the default KEMAR HRIR filters.
            """
            print("Loading default KEMAR filters.")
            self.current_sofa_file = "KEMAR_HRTF_FFComp.sofa"

            try:
                data_4 = xtc.extract_4_ir_sofa(self.current_sofa_file, left_az=-30.0, right_az=30.0)
                H_LL = data_4["H_LL"]
                H_LR = data_4["H_LR"]
                H_RL = data_4["H_RL"]
                H_RR = data_4["H_RR"]
                self.samplerate = data_4["samplerate"]

                self.f11_time, self.f12_time, self.f21_time, self.f22_time = xtc.generate_xtc_filters_mimo(
                    H_LL, H_LR, H_RL, H_RR,
                    samplerate=self.samplerate,
                    regularization=self.regularization
                )
                print("Default filters loaded successfully.")
            except Exception as e:
                print(f"Error loading default filters: {e}")
                
        def generate_hrir_and_reload_filters(self):
            """
            Generate a new HRIR SOFA file for the current head position and reload the filters
            """
            print("Generating HRIRs for the current head position...")

            left_ear = head_position + np.array([-ear_offset / 2, 0.0])
            right_ear = head_position + np.array([ear_offset / 2, 0.0])

            dist_L_L = np.linalg.norm(original_speaker_positions[0] - left_ear)
            dist_L_R = np.linalg.norm(original_speaker_positions[0] - right_ear)
            dist_R_L = np.linalg.norm(original_speaker_positions[1] - left_ear)
            dist_R_R = np.linalg.norm(original_speaker_positions[1] - right_ear)

            d_L_L = int(dist_L_L / c * self.samplerate)
            d_L_R = int(dist_L_R / c * self.samplerate)
            d_R_L = int(dist_R_L / c * self.samplerate)
            d_R_R = int(dist_R_R / c * self.samplerate)

            ir_len = 1024
            H_LL = np.zeros(ir_len)
            H_LR = np.zeros(ir_len)
            H_RL = np.zeros(ir_len)
            H_RR = np.zeros(ir_len)

            if d_L_L < ir_len:
                H_LL[d_L_L] = 1.0
            if d_L_R < ir_len:
                H_LR[d_L_R] = 1.0
            if d_R_L < ir_len:
                H_RL[d_R_L] = 1.0
            if d_R_R < ir_len:
                H_RR[d_R_R] = 1.0

            sofa_filename = "Generated_HRIR.sofa"
            xtc.save_to_sofa(sofa_filename, H_LL, H_LR, H_RL, H_RR, self.samplerate)

            self.current_sofa_file = sofa_filename
            print(f"Generated HRIRs saved to {sofa_filename}, reloading filters...")

            self.reload_filters_with_current_regularization()

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

            # Recompute the 4 lines for crosstalk
            # Using the MIMO filters: self.fXX_time
            lines = self.compute_transfer_functions_mimo_local()
            for i, (freqs, mags) in enumerate(lines):
                self.tf_curves[i].setData(freqs, mags)

        def compute_transfer_functions_mimo_local(self):
            """
            Exactly as compute_transfer_functions_mimo,
            but uses self.f11_time, etc. from the class
            rather than old global variables.
            """
            sig_len = 1024
            earL_imp = np.zeros(sig_len)
            earL_imp[0] = 1.0
            earR_imp = np.zeros(sig_len)
            earR_imp[0] = 1.0

            netL_Limp, netR_Limp = self.apply_xtc_filters_mimo_local(head_position, earL_imp, np.zeros_like(earL_imp))
            netL_Rimp, netR_Rimp = self.apply_xtc_filters_mimo_local(head_position, np.zeros_like(earR_imp), earR_imp)

            half = sig_len//2
            def fft_mag_db(sig):
                mag = np.abs(fft(sig))[:half]
                freqs = np.fft.fftfreq(sig_len, 1*self.samplerate)[:half]
                mag_db = 20*np.log10(mag / (mag.max()+1e-12) + 1e-12)
                return freqs, mag_db

            lines = []
            f_LL, m_LL = fft_mag_db(netL_Limp)
            lines.append((f_LL, m_LL))

            f_LR, m_LR = fft_mag_db(netR_Limp)
            lines.append((f_LR, m_LR))

            f_RL, m_RL = fft_mag_db(netL_Rimp)
            lines.append((f_RL, m_RL))

            f_RR, m_RR = fft_mag_db(netR_Rimp)
            lines.append((f_RR, m_RR))
            return lines

        def apply_xtc_filters_mimo_local(self, head_pos, earL, earR):
            """
            Same logic as apply_xtc_filters_mimo, but uses self.f11_time, etc.
            """
            # Convolve ear signals with the filters
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
                head_position[0] = mouse_real[0]
                head_position[1] = mouse_real[1]
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
    main_win = MainWindow()
    main_win.show()
    app.exec()

if __name__ == "__main__":
    simulate_xtc()