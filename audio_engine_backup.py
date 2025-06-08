from scipy.signal import butter, sosfilt, sosfilt_zi, oaconvolve
from scipy.signal import firwin
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
        self.bypass_buffer = np.zeros((0, 2), dtype=np.float32)
        self.streams = []
        self.levels = [0.0] * 4  # In L, In R, Out L, Out R
        self.latest_binaural_block = None
        self.latest_output_block = None

        self.xtc_last_end = np.zeros(2, dtype=np.float32)
        self.last_input_block = None   # For detecting repeated input blocks
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_meters)
        self.timer.start(30)
        self.lp_delay_samples = 391
        # Initialize buffers and counters for XTC alignment
        self.last_output_end = np.zeros(2, dtype=np.float32)
        self.fft_state = {}
        # Removed hp_buffer and xtc_block_count; no alignment FIFO needed for simple HP->XTC
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
                blocksize=1024
            )
            self.binaural_stream.start()
            print(f"Binaural stream blocksize: {self.binaural_stream.blocksize}")
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
                callback=self._output_callback,
                blocksize=1024
            )
            self.out_stream.start()
            print(f"Output stream blocksize: {self.out_stream.blocksize}")
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
        print(f"_apply_crossover_filters: input_buffer.shape={input_buffer.shape}")
        sig_lp_L, self.zlp_L = sosfilt(self.sos_lp, input_buffer[:, 0], zi=self.zlp_L)
        sig_lp_R, self.zlp_R = sosfilt(self.sos_lp, input_buffer[:, 1], zi=self.zlp_R)
        sig_hp_L, self.zhp_L = sosfilt(self.sos_hp, input_buffer[:, 0], zi=self.zhp_L)
        sig_hp_R, self.zhp_R = sosfilt(self.sos_hp, input_buffer[:, 1], zi=self.zhp_R)
        print(f"  pre-roll LP shapes: L={sig_lp_L.shape}, R={sig_lp_R.shape}; HP shapes: L={sig_hp_L.shape}, R={sig_hp_R.shape}")
        # Apply delay to LPF signals for crossover alignment
        sig_lp_L = np.roll(sig_lp_L, self.lp_delay_samples)
        sig_lp_R = np.roll(sig_lp_R, self.lp_delay_samples)
        print(f"  post-roll LP shapes: L={sig_lp_L.shape}, R={sig_lp_R.shape}")
        return (sig_lp_L, sig_lp_R), (sig_hp_L, sig_hp_R)

    def _apply_xtc_filters(self, high_passed_signals, frames):
        """Apply XTC filters using overlap-save convolution (no cross-block alignment)."""
        left, right = high_passed_signals
        print(f"Apply XTC: left.shape={left.shape}, right.shape={right.shape}, filter length M={len(self.f11) if self.f11 is not None else None}, initial frames parameter={frames}")
        N = left.shape[0]  # input block size
        # Check that all filters are not None
        if any(f is None for f in [self.f11, self.f12, self.f21, self.f22]):
            print("Warning: One or more XTC filters are not loaded.")
            return np.zeros(N), np.zeros(N)
        M = len(self.f11)
        fft_len = 2 ** int(np.ceil(np.log2(M + N - 1)))

        prev_l = self.fft_state.get('prev_l')
        prev_r = self.fft_state.get('prev_r')
        print(f"XTC state before block: prev_l length={prev_l.shape[0] if isinstance(prev_l, np.ndarray) else 'unset'}, prev_r length={prev_r.shape[0] if isinstance(prev_r, np.ndarray) else 'unset'}")

        # Setup state if not yet present
        if not self.fft_state:
            for ch in ['l', 'r']:
                self.fft_state[f'prev_{ch}'] = np.zeros(M-1)
            self.fft_state['F11'] = np.fft.fft(self.f11, fft_len)
            self.fft_state['F12'] = np.fft.fft(self.f12, fft_len)
            self.fft_state['F21'] = np.fft.fft(self.f21, fft_len)
            self.fft_state['F22'] = np.fft.fft(self.f22, fft_len)
            print(f"Initialized fft_state keys: {list(self.fft_state.keys())}")
        def convolve_overlap_save(x, h_fft, prev):
            x_full = np.concatenate([prev, x])
            print(f"convolve_overlap_save: prev length={prev.shape[0]}, x length={x.shape[0]}, x_full length={x_full.shape[0]}, fft_len={fft_len}")
            X = np.fft.fft(x_full, fft_len)
            Y = np.fft.ifft(X * h_fft).real
            print(f"Y total length={Y.shape[0]}, extracting from index {M-1} to {M-1+N}")
            # Safety assertion for correct output shape
            assert Y[M-1:M-1+N].shape[0] == N
            print(f"convolve_overlap_save output length={Y[M-1:M-1+N].shape[0]}")
            return Y[M-1:M-1+N], x_full[-(M-1):]

        # Preserve old prev buffers for left and right
        prev_l_old = self.fft_state['prev_l']
        prev_r_old = self.fft_state['prev_r']
        # Convolve left input with F11 and F21 using same prev_l_old
        sig_L1, new_prev_l = convolve_overlap_save(left, self.fft_state['F11'], prev_l_old)
        sig_R1, _ = convolve_overlap_save(left, self.fft_state['F21'], prev_l_old)
        # Convolve right input with F12 and F22 using same prev_r_old
        sig_L2, new_prev_r = convolve_overlap_save(right, self.fft_state['F12'], prev_r_old)
        sig_R2, _ = convolve_overlap_save(right, self.fft_state['F22'], prev_r_old)
        # Update state buffers once
        self.fft_state['prev_l'] = new_prev_l
        self.fft_state['prev_r'] = new_prev_r

        gain = 10 ** (-20 / 20)
        # Combine HP channels and return immediately (no cross-block alignment)
        sig_L = (sig_L1 + sig_L2) * gain
        sig_R = (sig_R1 + sig_R2) * gain
        return sig_L, sig_R

    def _reconstruct_output_signal(self, low_passed_signals, xtc_filtered_signals):
        """Sum low-passed and XTC-filtered signals, stack to stereo output, and clip."""
        print(f"_reconstruct_output_signal: LP shapes: {low_passed_signals[0].shape}, {low_passed_signals[1].shape}; XTC shapes: {xtc_filtered_signals[0].shape}, {xtc_filtered_signals[1].shape}")
        out_L = low_passed_signals[0] + xtc_filtered_signals[0]
        out_R = low_passed_signals[1] + xtc_filtered_signals[1]
        output = np.vstack((out_L, out_R)).T
        np.clip(output, -1.0, 1.0, out=output)
        print(f"  created stereo output.shape={output.shape}")
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
            print(f"_binaural_callback: frames={frames}, indata.shape={indata.shape}")
        self._copy_input_buffer(indata)
        # If bypass, queue input blocks for output
        if self.bypass:
            self.bypass_buffer = np.vstack((self.bypass_buffer, indata.copy()))
        print(f"_binaural_callback: frames={frames}, indata.shape={indata.shape}")
        rms = np.sqrt(np.mean(indata ** 2, axis=0))
        self.levels[0] = float(rms[0])
        self.levels[1] = float(rms[1])
        # Meter updates are driven by QTimer

    def _output_callback(self, outdata, frames, time, status):
        print(f"_output_callback: frames={frames}, outdata.shape={outdata.shape}")
        if self.latest_binaural_block is not None:
            print(f"latest_binaural_block.shape={self.latest_binaural_block.shape}")
        else:
            print("latest_binaural_block is None")
        if self.latest_binaural_block is not None and self.latest_binaural_block.shape[0] != frames:
            print(f"*** MISMATCH WARNING: latest_binaural_block length {self.latest_binaural_block.shape[0]} != frames {frames}")
        outdata[:] = np.zeros((frames, 2), dtype=np.float32)
        if self.latest_binaural_block is None:
            return
        try:
            # Bypass if set, or if any XTC filter is None
            if self.bypass or any(f is None for f in [self.f11, self.f12, self.f21, self.f22]):
                print(f"Bypass branch: outputting from bypass_buffer (size={self.bypass_buffer.shape[0]})")
                # If enough samples in bypass_buffer, pop the earliest 'frames'
                if self.bypass_buffer.shape[0] >= frames:
                    to_write = self.bypass_buffer[:frames]
                    self.bypass_buffer = self.bypass_buffer[frames:]
                else:
                    # Not enough data, pad with zeros
                    needed = frames - self.bypass_buffer.shape[0]
                    pad = np.zeros((needed, 2), dtype=self.bypass_buffer.dtype)
                    to_write = np.vstack((self.bypass_buffer, pad))
                    self.bypass_buffer = np.zeros((0, 2), dtype=np.float32)
                outdata[:] = to_write
                rms_out = np.sqrt(np.mean(outdata ** 2, axis=0))
                self.levels[2] = float(rms_out[0])
                self.levels[3] = float(rms_out[1])
                return
            else:
                input_buffer = self.latest_binaural_block
                # Detect repeated input block as before
                if self.last_input_block is not None and np.array_equal(input_buffer, self.last_input_block):
                    print("WARNING: repeated input_buffer block (possible input/output rate mismatch)")
                self.last_input_block = input_buffer.copy()
                # Use raw channels directly (HP integrated in filter FFTs)
                left = input_buffer[:, 0]
                right = input_buffer[:, 1]
                xtc_L, xtc_R = self._apply_xtc_filters((left, right), frames)
                # Reconstruct output as stereo from XTC only
                output = np.vstack((xtc_L, xtc_R)).T
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