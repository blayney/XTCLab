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

    # ---------------------------------------------
    # Setup geometry and scale factors
    # ---------------------------------------------
    distance = 1.5 # from centre of perfectly circular head to speakers
    left_az_deg = -30.0
    right_az_deg = 30.0

    head_position = np.array([0.0, 1.0])
    ear_offset = 0.15
    scale_factor = 10.0
    c = 343.0 # speed of sound

    # Speakers at ±30 degrees azimuth
    xL = head_position[0] + distance * sin(radians(left_az_deg))
    yL = head_position[1] + distance * cos(radians(left_az_deg))

    xR = head_position[0] + distance * sin(radians(right_az_deg))
    yR = head_position[1] + distance * cos(radians(right_az_deg))

    original_speaker_positions = np.array([[xL, yL],
                                           [xR, yR]])
    
    # we hope this is where the trough of the energy distribution will be found.
    ideal_position = np.array([0.0, 1.0])  

    # ---------------------------------------------
    # Load 4 IRs and build MIMO XTC filters
    # ---------------------------------------------
    print("Extracting HRTFs, building 2×2 MIMO XTC filters...")
    data_4 = xtc.extract_4_ir_sofa("KEMAR_HRTF_FFComp.sofa",
                                   left_az=-30.0,
                                   right_az=30.0)
    H_LL = data_4["H_LL"]
    H_LR = data_4["H_LR"]
    H_RL = data_4["H_RL"]
    H_RR = data_4["H_RR"]
    samplerate = data_4["samplerate"]

    f11_time, f12_time, f21_time, f22_time = xtc.generate_xtc_filters_mimo(
        H_LL, H_LR, H_RL, H_RR,
        samplerate=samplerate,
        regularization=0.01
    )

    print("XTC filters built\n")

    # ---------------------------------------------
    # MIMO apply function
    # ---------------------------------------------
    def apply_xtc_filters_mimo(head_pos, ear_L_signal, ear_R_signal):
        """
          speaker_L(t) = f11_time*ear_L + f12_time*ear_R
          speaker_R(t) = f21_time*ear_L + f22_time*ear_R
        Then delayed to each ear.
        Returns: (net_left_ear, net_right_ear)
        """
        # 1) Convolve ear signals with filter matrix
        spkL_from_left  = np.convolve(ear_L_signal, f11_time, mode='same')
        spkL_from_right = np.convolve(ear_R_signal, f12_time, mode='same')
        speaker_L = spkL_from_left + spkL_from_right

        spkR_from_left  = np.convolve(ear_L_signal, f21_time, mode='same')
        spkR_from_right = np.convolve(ear_R_signal, f22_time, mode='same')
        speaker_R = spkR_from_left + spkR_from_right

        # 2) Time-of-flight to each ear
        left_ear_pos  = head_pos + np.array([-ear_offset/2, 0.0])
        right_ear_pos = head_pos + np.array([ ear_offset/2, 0.0])

        dist_left_ear  = np.linalg.norm(original_speaker_positions - left_ear_pos, axis=1)
        dist_right_ear = np.linalg.norm(original_speaker_positions - right_ear_pos, axis=1)
        delay_left_samps  = (dist_left_ear  / c * samplerate).astype(int)
        delay_right_samps = (dist_right_ear / c * samplerate).astype(int)

        # speaker0->LeftEar, speaker1->LeftEar
        spk0_left = np.roll(speaker_L, delay_left_samps[0])
        spk1_left = np.roll(speaker_R, delay_left_samps[1])
        net_left_ear = spk0_left + spk1_left

        # speaker0->RightEar, speaker1->RightEar
        spk0_right = np.roll(speaker_L, delay_right_samps[0])
        spk1_right = np.roll(speaker_R, delay_right_samps[1])
        net_right_ear = spk0_right + spk1_right

        return net_left_ear, net_right_ear

    # ---------------------------------------------
    # FFTs for transfer functions
    # ---------------------------------------------
    def compute_transfer_functions_mimo():
        """
        We do two impulses:
          (A) leftEar=Impulse, rightEar=0
          (B) leftEar=0,       rightEar=Impulse
        Then measure netLeftEar & netRightEar in each scenario.
        Return 4 sets of (freqs, mag_db).
        """
        sig_len = 1024
        earL_imp = np.zeros(sig_len)
        earL_imp[0] = 1.0
        earR_imp = np.zeros(sig_len)
        earR_imp[0] = 1.0

        # (A) leftEar=Imp
        netL_Limp, netR_Limp = apply_xtc_filters_mimo(head_position, earL_imp, np.zeros_like(earL_imp))
        # (B) rightEar=Imp
        netL_Rimp, netR_Rimp = apply_xtc_filters_mimo(head_position, np.zeros_like(earR_imp), earR_imp)

        half = sig_len//2
        def fft_mag_db(sig):
            mag = np.abs(fft(sig))[:half]
            freqs = np.fft.fftfreq(sig_len, 1/samplerate)[:half]
            mag_db = 20*np.log10(mag / (mag.max()+1e-12) + 1e-12)
            return freqs, mag_db

        # Build list: (EarLImp->LeftEar), (EarLImp->RightEar), (EarRImp->LeftEar), (EarRImp->RightEar)
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

    # ---------------------------------------------
    # Rendering code
    # ---------------------------------------------
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("XTC 2×2 MIMO Simulation + Energy Demo")
            self.resize(1200, 800)

            # Dist. overlay turned off initially
            self.energy_distribution_enabled = False

            # FFTs (dock / floating window)
            self.fftDock = QDockWidget("Crosstalk Demo FFTs", self)
            features = (QDockWidget.DockWidgetFeature.DockWidgetMovable |
                        QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.fftDock.setFeatures(features)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.fftDock)

            fft_container = QtWidgets.QWidget()
            fft_layout = QtWidgets.QGridLayout(fft_container)
            self.fftDock.setWidget(fft_container)

            # subplots
            self.tf_plots = []
            self.tf_curves = []
            plot_titles = [
                "EarL → netLeftEar",
                "EarL → netRightEar",
                "EarR → netLeftEar",
                "EarR → netRightEar"
            ]
            for i, t_str in enumerate(plot_titles):
                pw = pg.PlotWidget(title=t_str)
                pw.setLabel("bottom", "Frequency (Hz)")
                pw.setLabel("left", "Magnitude (dB)")
                pw.setYRange(-60, 10)
                curve = pw.plot(pen="b")
                self.tf_plots.append(pw)
                self.tf_curves.append(curve)
                fft_layout.addWidget(pw, i//2, i%2)

            # Top-down Plot
            self.plot = DraggablePlot(title="2D Top-Down View")
            self.plot.setAspectLocked(True)
            # We'll add an ImageItem for energy in background
            self.energy_item = pg.ImageItem()
            self.energy_item.setZValue(-100)  # behind lines
            self.energy_item.setVisible(False)
            self.plot.addItem(self.energy_item)  # keep it always in scene

            # Speaker->Ear lines
            self.speaker_ear_lines = []
            self.speaker_ear_labels = []
            line_pen = pg.mkPen(color="gray", width=1, dash=[2,4])
            for _ in range(4):
                line_item = pg.PlotCurveItem(pen=line_pen)
                self.speaker_ear_lines.append(line_item)
                self.plot.addItem(line_item)
                txt = pg.TextItem(anchor=(0.5, 1.0))
                txt.setColor("gray")
                self.speaker_ear_labels.append(txt)
                self.plot.addItem(txt)

            center_widget = QtWidgets.QWidget()
            center_layout = QtWidgets.QVBoxLayout(center_widget)
            self.setCentralWidget(center_widget)
            center_layout.addWidget(self.plot)

            # Buttons
            reset_button = QtWidgets.QPushButton("Reset Head Position")
            reset_button.clicked.connect(self.reset_head)
            center_layout.addWidget(reset_button)

            self.energy_checkbox = QtWidgets.QCheckBox("Show Energy Distribution")
            self.energy_checkbox.stateChanged.connect(self.toggle_energy_distribution)
            center_layout.addWidget(self.energy_checkbox)

            # Markers
            spk_x = original_speaker_positions[:,0]*scale_factor
            spk_y = original_speaker_positions[:,1]*scale_factor
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
            ideal_x = ideal_position[0]*scale_factor
            ideal_y = ideal_position[1]*scale_factor
            self.ideal_marker.setData([ideal_x],[ideal_y])
            self.plot.addItem(self.ideal_marker)

            self.auto_range_topdown_plot()
            self.update_plot()

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
            Compute the degree of crosstalk cancellation on a 4x4 meter grid around the (0,1) region.
            The map is computed based on the mean contralateral cancellation at each point in space.
            """
            grid_size = 400
            xvals = np.linspace(-2, 2, grid_size)  # X-axis positions
            yvals = np.linspace(-2, 2, grid_size)  # Y-axis positions
            energy_map = np.zeros((grid_size, grid_size), dtype=np.float32)

            for j, yval in enumerate(yvals):
                for i, xval in enumerate(xvals):
                    # Test position in real coordinates
                    test_pos = np.array([xval, yval])

                    # Compute ear positions
                    left_ear = test_pos + np.array([-ear_offset / 2, 0.0])
                    right_ear = test_pos + np.array([ear_offset / 2, 0.0])

                    # Distances from each speaker to each ear
                    dist_leftSpk_to_leftEar = np.linalg.norm(original_speaker_positions[0] - left_ear)
                    dist_leftSpk_to_rightEar = np.linalg.norm(original_speaker_positions[0] - right_ear)
                    dist_rightSpk_to_leftEar = np.linalg.norm(original_speaker_positions[1] - left_ear)
                    dist_rightSpk_to_rightEar = np.linalg.norm(original_speaker_positions[1] - right_ear)

                    # Calculate delays in samples for each speaker-to-ear path
                    delay_leftSpk_to_leftEar = int((dist_leftSpk_to_leftEar / c) * samplerate)
                    delay_leftSpk_to_rightEar = int((dist_leftSpk_to_rightEar / c) * samplerate)
                    delay_rightSpk_to_leftEar = int((dist_rightSpk_to_leftEar / c) * samplerate)
                    delay_rightSpk_to_rightEar = int((dist_rightSpk_to_rightEar / c) * samplerate)

                    # Simulate cancellation using impulses
                    impulse = np.zeros(1024)
                    impulse[0] = 1.0  # Unit impulse
                    left_speaker_signal = np.roll(impulse, delay_leftSpk_to_leftEar) + np.roll(impulse, delay_leftSpk_to_rightEar)
                    right_speaker_signal = np.roll(impulse, delay_rightSpk_to_leftEar) + np.roll(impulse, delay_rightSpk_to_rightEar)

                    # Apply MIMO filters
                    spkL_from_left = np.convolve(left_speaker_signal, f11_time, mode='same')
                    spkL_from_right = np.convolve(right_speaker_signal, f12_time, mode='same')
                    spkR_from_left = np.convolve(left_speaker_signal, f21_time, mode='same')
                    spkR_from_right = np.convolve(right_speaker_signal, f22_time, mode='same')

                    # Net ear signals after MIMO processing
                    net_left_ear = spkL_from_left + spkR_from_left
                    net_right_ear = spkL_from_right + spkR_from_right

                    # Compute the power of contralateral components
                    contralateral_cancellation_left = np.sum(net_right_ear**2) / np.sum(impulse**2)
                    contralateral_cancellation_right = np.sum(net_left_ear**2) / np.sum(impulse**2)

                    # Average contralateral cancellation
                    mean_cancellation = (contralateral_cancellation_left + contralateral_cancellation_right) / 2

                    # Store in the energy map
                    energy_map[j, i] = -10 * np.log10(mean_cancellation + 1e-12)  # Convert to dB

            return xvals, yvals, energy_map    
            
        def update_plot(self):
            print("Updating plot...")

            # If energy distribution is on, show the image in the background
            if self.energy_distribution_enabled:
                xvals, yvals, energy_map = self.compute_energy_distribution()
                # Convert real coords to scaled coords
                # We'll place the image from x=-2..2 => scaled: -20..+20 if scale_factor=10
                x_min = xvals[0]*scale_factor
                y_min = yvals[0]*scale_factor
                width = (xvals[-1]-xvals[0])*scale_factor
                height= (yvals[-1]-yvals[0])*scale_factor

                # Show the map
                self.energy_item.setImage(energy_map.T)  # transpose so X->cols, Y->rows
                self.energy_item.setRect(pg.QtCore.QRectF(x_min, y_min, width, height))
                self.energy_item.setVisible(True)
            else:
                self.energy_item.setVisible(False)

            # Update top-down items (head, ears, lines, etc.)
            hx = head_position[0]*scale_factor
            hy = head_position[1]*scale_factor
            self.head_circle.setData([hx],[hy])

            left_ear = head_position + np.array([-ear_offset/2, 0.0])
            right_ear= head_position + np.array([ ear_offset/2, 0.0])
            self.ear_dots.setData(
                [left_ear[0]*scale_factor, right_ear[0]*scale_factor],
                [left_ear[1]*scale_factor, right_ear[1]*scale_factor]
            )

            # 4 lines: spk[0]->LeftEar, spk[0]->RightEar, spk[1]->LeftEar, spk[1]->RightEar
            connections = [
                (original_speaker_positions[0], left_ear),
                (original_speaker_positions[0], right_ear),
                (original_speaker_positions[1], left_ear),
                (original_speaker_positions[1], right_ear)
            ]
            for i, (spk, ear) in enumerate(connections):
                spk_scaled = spk*scale_factor
                ear_scaled = ear*scale_factor
                self.speaker_ear_lines[i].setData(
                    x=[spk_scaled[0], ear_scaled[0]],
                    y=[spk_scaled[1], ear_scaled[1]]
                )
                dist_m = np.linalg.norm(spk-ear)
                delay_ms = (dist_m/c)*1000
                mid = (spk+ear)/2.0
                mid_scaled = mid*scale_factor
                self.speaker_ear_labels[i].setPos(mid_scaled[0], mid_scaled[1])
                self.speaker_ear_labels[i].setText(f"{dist_m:.2f} m\n{delay_ms:.2f} ms")

            # Update the 4 FFT lines for the crosstalk scenarios
            lines = compute_transfer_functions_mimo()
            for i, (freqs, mags) in enumerate(lines):
                self.tf_curves[i].setData(freqs, mags)

        def reset_head(self):
            head_position[:] = ideal_position
            self.update_plot()

    # ---------------------------------------------
    # Handling for objects that can be dragged around
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