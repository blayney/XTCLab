from scipy.signal import butter, sosfilt, sosfilt_zi, oaconvolve
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QTimer

class AudioEngine:
    def __init__(self, config, meter_callback=None, f11=None, f12=None, f21=None, f22=None, bypass=False):
        self.config = config
        self.meter_callback = meter_callback
        self.streams = []
        self.levels = [0.0] * 4  # In L, In R, Out L, Out R
        self.f11 = f11
        self.f12 = f12
        self.f21 = f21
        self.f22 = f22
        self.latest_binaural_block = None
        self.latest_output_block = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_meters)
        self.timer.start(30)
        print("AudioEngine initialized with config:", self.config)
        self.bypass = bypass
        # Design causal IIR crossover filters at 80 Hz (4th-order Butterworth)
        fs = 48000.0
        self.sos_hp = butter(4, 80.0, btype='high', fs=fs, output='sos')
        self.sos_lp = butter(4, 80.0, btype='low',  fs=fs, output='sos')
        # Initialize per-channel filter state for streaming sosfilt
        self.zlp_L = sosfilt_zi(self.sos_lp)
        self.zlp_R = sosfilt_zi(self.sos_lp)
        self.zhp_L = sosfilt_zi(self.sos_hp)
        self.zhp_R = sosfilt_zi(self.sos_hp)
 
        self._setup_streams()

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

    def _binaural_callback(self, indata, frames, time, status):
        if status:
            print("Binaural input stream status:", status)
        # print("Received audio in binaural callback.")
        # print(f"Binaural callback: indata shape {indata.shape}, RMS L={np.sqrt(np.mean(indata[:,0]**2)):.4f}")
        self.latest_binaural_block = indata.copy()
        rms = np.sqrt(np.mean(indata**2, axis=0))
        self.levels[0] = float(rms[0])
        self.levels[1] = float(rms[1])
        # Removed immediate meter update; QTimer will drive updates

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

    def _output_callback(self, outdata, frames, time, status):
        outdata[:] = np.zeros((frames, 2), dtype=np.float32)

        if self.latest_binaural_block is None:
            return

        try:
            if self.bypass:
                outdata[:] = self.latest_binaural_block.copy()  # Direct passthrough (bypass mode)
                # Store the bypassed output block for FFT display
                self.latest_output_block = outdata.copy()
                # Compute real output levels from bypassed data
                rms_out = np.sqrt(np.mean(outdata**2, axis=0))
                self.levels[2] = float(rms_out[0])
                self.levels[3] = float(rms_out[1])
                # Removed immediate meter update; QTimer will drive updates
            else:
                # 1) XTC‑filtered full‑band signals
                left = self.latest_binaural_block[:, 0]
                right = self.latest_binaural_block[:, 1]
                # Partitioned overlap-add convolution
                sig_L = oaconvolve(left, self.f11, mode='same') + oaconvolve(right, self.f12, mode='same')
                sig_R = oaconvolve(left, self.f21, mode='same') + oaconvolve(right, self.f22, mode='same')

                # 2) Causal IIR crossover: low-pass direct + high-pass filtered, streaming
                sig_lp_L, self.zlp_L = sosfilt(self.sos_lp, left,  zi=self.zlp_L)
                sig_lp_R, self.zlp_R = sosfilt(self.sos_lp, right, zi=self.zlp_R)
                sig_hp_L, self.zhp_L = sosfilt(self.sos_hp, sig_L,   zi=self.zhp_L)
                sig_hp_R, self.zhp_R = sosfilt(self.sos_hp, sig_R,   zi=self.zhp_R)

                # 3) Ensure each branch matches block length
                out_L = sig_lp_L[:frames] if len(sig_lp_L) >= frames else np.pad(sig_lp_L, (0, frames-len(sig_lp_L)))
                out_R = sig_lp_R[:frames] if len(sig_lp_R) >= frames else np.pad(sig_lp_R, (0, frames-len(sig_lp_R)))
                out_L += sig_hp_L[:frames] if len(sig_hp_L) >= frames else np.pad(sig_hp_L, (0, frames-len(sig_hp_L)))
                out_R += sig_hp_R[:frames] if len(sig_hp_R) >= frames else np.pad(sig_hp_R, (0, frames-len(sig_hp_R)))

                # 4) Prevent clipping
                np.clip(np.vstack((out_L, out_R)).T, -1.0, 1.0, out=outdata)

                # 5) Store for FFT display and metering
                self.latest_output_block = outdata.copy()
                self.levels[2] = np.sqrt(np.mean(outdata[:,0]**2))
                self.levels[3] = np.sqrt(np.mean(outdata[:,1]**2))
                # Removed immediate meter update; QTimer will drive updates
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