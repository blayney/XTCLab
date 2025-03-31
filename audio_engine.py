import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QTimer

class AudioEngine:
    def __init__(self, config, meter_callback=None, f11=None, f12=None, f21=None, f22=None, bypass=False):
        self.config = config
        self.meter_callback = meter_callback
        self.streams = []
        self.levels = [0.0] * 5  # In L, In R, Mic, Out L, Out R
        self.f11 = f11
        self.f12 = f12
        self.f21 = f21
        self.f22 = f22
        self.latest_binaural_block = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_meters)
        self.timer.start(30)
        print("AudioEngine initialized with config:", self.config)
        self.bypass = bypass
 
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
            print("Testing BlackHole with sd.rec()...")
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
            print("Starting mic input stream on:", self.config["measurement_device"])
            index = self._find_device_index(self.config["measurement_device"])
            self.mic_stream = sd.InputStream(
                device=index,
                channels=1,
                dtype='float32',
                samplerate=48000,
                callback=self._mic_callback,
                latency='low',
                blocksize=512
            )
            self.mic_stream.start()
            self.streams.append(self.mic_stream)

            print("Starting output stream on:", self.config["playback_device"])
            index = self._find_device_index(self.config["playback_device"])
            self.out_stream = sd.OutputStream(
                device=index,
                channels=2,
                samplerate=48000,
                callback=self._output_callback,
            )
            self.out_stream.start()
            self.streams.append(self.out_stream)
            print("Audio streams started.")

        except Exception as e:
            print("Error setting up streams:", e)

    def _binaural_callback(self, indata, frames, time, status):
        if status:
            print("Binaural input stream status:", status)
        print("Received audio in binaural callback.")
        print(f"Binaural callback: indata shape {indata.shape}, RMS L={np.sqrt(np.mean(indata[:,0]**2)):.4f}")
        self.latest_binaural_block = indata.copy()
        rms = np.sqrt(np.mean(indata**2, axis=0))
        self.levels[0] = float(rms[0])
        self.levels[1] = float(rms[1])
    def _mic_callback(self, indata, frames, time, status):
        if status:
            print("Mic input stream status:", status)
        print(f"Mic callback: indata shape {indata.shape}, RMS={np.sqrt(np.mean(indata**2)):.4f}")
        rms = np.sqrt(np.mean(indata**2))
        self.levels[2] = float(rms)

    def _update_meters(self):
        normed = []
        for i in range(5):
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
                # Direct passthrough (bypass mode)
                sig_L = self.latest_binaural_block[:, 0]
                sig_R = self.latest_binaural_block[:, 1]
            else:
                left = self.latest_binaural_block[:, 0]
                right = self.latest_binaural_block[:, 1]
                sig_L = np.convolve(left, self.f11, mode='same') + np.convolve(right, self.f12, mode='same')
                sig_R = np.convolve(left, self.f21, mode='same') + np.convolve(right, self.f22, mode='same')

                sig_L = sig_L[:frames] if len(sig_L) >= frames else np.pad(sig_L, (0, frames - len(sig_L)))
                sig_R = sig_R[:frames] if len(sig_R) >= frames else np.pad(sig_R, (0, frames - len(sig_R)))
                outdata[:, 0] = sig_L
                outdata[:, 1] = sig_R

                # Optional metering for output (not yet implemented)
                self.levels[3] = np.sqrt(np.mean(sig_L**2))
                self.levels[4] = np.sqrt(np.mean(sig_R**2))

        except Exception as e:
            print("Error in output processing:", e)