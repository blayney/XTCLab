from scipy.signal import butter, sosfilt, sosfilt_zi, oaconvolve
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QTimer

class AudioEngine:
    def __init__(self, config, meter_callback=None, f11=None, f12=None, f21=None, f22=None, bypass=False):
        self.config = config
        self.meter_callback = meter_callback
        self.f11 = f11
        self.f12 = f12
        self.f21 = f21
        self.f22 = f22
        self.bypass = bypass
        self.streams = []
        self.levels = [0.0] * 4  # In L, In R, Out L, Out R
        self.latest_binaural_block = None
        self.latest_output_block = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_meters)
        self.timer.start(30)
        self.lp_delay_samples = 391
        print("AudioEngine initialized with config:", self.config)
        self._setup_filters()
        self._setup_streams()

    def _setup_filters(self):
        """Design and initialize crossover filters and filter states, and compute LPF delay for alignment."""
        fs = 48000.0
        self.sos_hp = butter(8, 80.0, btype='high', fs=fs, output='sos')
        self.sos_lp = butter(8, 80.0, btype='low', fs=fs, output='sos')
        self.zlp_L = sosfilt_zi(self.sos_lp)
        self.zlp_R = sosfilt_zi(self.sos_lp)
        self.zhp_L = sosfilt_zi(self.sos_hp)
        self.zhp_R = sosfilt_zi(self.sos_hp)
        

    def _find_device_index(self, name):
        for i, dev in enumerate(sd.query_devices()):
            if dev["name"] == name:
                print(f"Resolved device '{name}' to index {i}")
                return i
        raise RuntimeError(f"Device '{name}' not found.")

    def _setup_streams(self):
        try:
            # Binaural input (2ch)
            print("Starting binaural input stream on:", self.config["binaural_device"])
            index = self._find_device_index(self.config["binaural_device"])
            self.binaural_stream = sd.InputStream(
                device=index,
                channels=2,
                dtype='float32',
                samplerate=48000,
                callback=self._binaural_callback,
                latency='low',
                blocksize=512
            )
            self.binaural_stream.start()
            self.streams.append(self.binaural_stream)

            # Mic input (1ch)
            # print("Starting mic input stream on:", self.config["measurement_device"])
            # index = self._find_device_index(self.config["measurement_device"])
            # self.mic_stream = sd.InputStream(
            #     device=index,
            #     channels=1,
            #     dtype='float32',
            #     samplerate=48000,
            #     callback=self._mic_callback,
            #     latency='low',
            #     blocksize=512
            # )
            # self.mic_stream.start()
            # self.streams.append(self.mic_stream)

            print("Starting output stream on:", self.config["playback_device"])
            index = self._find_device_index(self.config["playback_device"])
            self.out_stream = sd.OutputStream(
                device=index,
                channels=2,
                samplerate=48000,
                callback=self._output_callback
            )
            self.out_stream.start()
            self.streams.append(self.out_stream)
            print("Audio streams started.")

        except Exception as e:
            print("Error setting up streams:", e)

    def _copy_input_buffer(self, input_data):
        """Copy and store the latest binaural input block."""
        self.latest_binaural_block = input_data.copy()
        return self.latest_binaural_block

    def _apply_crossover_filters(self, input_buffer):
        """Apply crossover filters to input_buffer, return (lp_L, lp_R), (hp_L, hp_R)."""
        sig_lp_L, self.zlp_L = sosfilt(self.sos_lp, input_buffer[:, 0], zi=self.zlp_L)
        sig_lp_R, self.zlp_R = sosfilt(self.sos_lp, input_buffer[:, 1], zi=self.zlp_R)
        sig_hp_L, self.zhp_L = sosfilt(self.sos_hp, input_buffer[:, 0], zi=self.zhp_L)
        sig_hp_R, self.zhp_R = sosfilt(self.sos_hp, input_buffer[:, 1], zi=self.zhp_R)
        # Apply delay to LPF signals for crossover alignment
        sig_lp_L = np.roll(sig_lp_L, self.lp_delay_samples)
        sig_lp_R = np.roll(sig_lp_R, self.lp_delay_samples)
        return (sig_lp_L, sig_lp_R), (sig_hp_L, sig_hp_R)

    def _apply_xtc_filters(self, high_passed_signals):
        """Apply XTC filters to high-passed signals, return (sig_L, sig_R)."""
        left, right = high_passed_signals
        sig_L = oaconvolve(left, self.f11, mode='same') + oaconvolve(right, self.f12, mode='same')
        sig_R = oaconvolve(left, self.f21, mode='same') + oaconvolve(right, self.f22, mode='same')
        return sig_L, sig_R

    def _reconstruct_output_signal(self, low_passed_signals, xtc_filtered_signals):
        """Sum low-passed and XTC-filtered signals, stack to stereo output, and clip."""
        out_L = low_passed_signals[0] + xtc_filtered_signals[0]
        out_R = low_passed_signals[1] + xtc_filtered_signals[1]
        output = np.vstack((out_L, out_R)).T
        np.clip(output, -1.0, 1.0, out=output)
        return output

    def _dispatch_output_buffer(self, output_buffer, outdata):
        """Write output_buffer to outdata and store for FFT display."""
        outdata[:] = output_buffer
        self.latest_output_block = output_buffer.copy()

    def _mic_callback(self, indata, frames, time, status):
        if status:
            print("Mic input stream status:", status)
        # print(f"Mic callback: indata shape {indata.shape}, RMS={np.sqrt(np.mean(indata**2)):.4f}")
        rms = np.sqrt(np.mean(indata**2))
        #self.levels[2] = float(rms)

    def _update_meters(self):
        normed = []
        for i in range(4):
            db = 20 * np.log10(self.levels[i] + 1e-9)
            norm = min(max((db + 60) / 60.0, 0.0), 1.0)
            normed.append(norm)

        if self.meter_callback:
            if not hasattr(self, "_smoothed_levels"):
                self._smoothed_levels = normed
            else:
                alpha = 0.2  # smoothing factor
                self._smoothed_levels = [
                    alpha * new + (1 - alpha) * old
                    for new, old in zip(normed, self._smoothed_levels)
                ]
            self.meter_callback(self._smoothed_levels)

    def _binaural_callback(self, indata, frames, time, status):
        if status:
            print("Binaural input stream status:", status)
        self._copy_input_buffer(indata)
        rms = np.sqrt(np.mean(indata ** 2, axis=0))
        self.levels[0] = float(rms[0])
        self.levels[1] = float(rms[1])
        # Meter updates are driven by QTimer

    def _output_callback(self, outdata, frames, time, status):
        outdata[:] = np.zeros((frames, 2), dtype=np.float32)
        if self.latest_binaural_block is None:
            return
        try:
            if self.bypass:
                # Direct passthrough (bypass mode)
                self._dispatch_output_buffer(self.latest_binaural_block, outdata)
                rms_out = np.sqrt(np.mean(outdata ** 2, axis=0))
                self.levels[2] = float(rms_out[0])
                self.levels[3] = float(rms_out[1])
            else:
                # Modular pipeline:
                input_buffer = self.latest_binaural_block
                # (lp_L, lp_R), (hp_L, hp_R)
                (lp_L, lp_R), (hp_L, hp_R) = self._apply_crossover_filters(input_buffer)
                xtc_filtered = self._apply_xtc_filters((hp_L, hp_R))
                output = self._reconstruct_output_signal((lp_L, lp_R), xtc_filtered)
                # Ensure output matches required frame count
                if output.shape[0] < frames:
                    output = np.pad(output, ((0, frames - output.shape[0]), (0, 0)))
                elif output.shape[0] > frames:
                    output = output[:frames, :]
                self._dispatch_output_buffer(output, outdata)
                self.levels[2] = np.sqrt(np.mean(outdata[:, 0] ** 2))
                self.levels[3] = np.sqrt(np.mean(outdata[:, 1] ** 2))
        except Exception as e:
            print("Error in output processing:", e)

    def close(self):
        """Stop and close all audio streams."""
        for stream in getattr(self, "streams", []):
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self.streams = []

    def set_bypass(self, bypass: bool):
        """Enable or disable filter bypass without restarting streams."""
        self.bypass = bypass