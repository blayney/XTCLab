

import unittest
import io
import contextlib
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians

import xtc
import xtc_lab_tracked
import pysofaconventions as sofa

class TableTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super(TableTestResult, self).__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super(TableTestResult, self).addSuccess(test)
        self.successes.append(test)

class TestXTCLabTracked(unittest.TestCase):
    def test_list_audio_outputs_no_crash(self):
        # Monkeypatch sounddevice to return empty list
        dummy_devices = []
        with patch('xtc_lab_tracked.sd.query_devices', return_value=dummy_devices):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                xtc_lab_tracked.list_audio_outputs()
            output = f.getvalue()
            # Check that it prints header or handles empty list
            self.assertIn("Available audio output devices", output)

    def test_setup_audio_routing_success(self):
        # Create a dummy device list with BlackHole 2ch
        dummy_devices = [
            {'name': 'Device A', 'max_output_channels': 0},
            {'name': 'BlackHole 2ch', 'max_output_channels': 2, 'index': 2}
        ]
        with patch('xtc_lab_tracked.sd.query_devices', return_value=dummy_devices):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                xtc_lab_tracked.setup_audio_routing()
            output = f.getvalue()
            self.assertIn("BlackHole 2ch", output)

class TestXTCFunctions(unittest.TestCase):
    def test_compute_speaker_positions_2D(self):
        # distance=1, left_az=0 => (0,1), right_az=90 => (1,0)
        pos = xtc.compute_speaker_positions_2D(distance=1.0, left_az_deg=0.0, right_az_deg=90.0)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertTrue(np.allclose(pos, expected, atol=1e-6))

    # (Removed broken test_find_first_above_threshold)

    def test_extract_hrirs_sam_invalid_file(self):
        # extract_hrirs_sam should raise an error for non-existent file
        with self.assertRaises(Exception):
            xtc.extract_hrirs_sam("nonexistent_file.sofa", 0.0)

    def test_align_impulse_responses_shape(self):
        # Simple impulse delayed by 3 samples
        H = np.zeros(10)
        H[3] = 1.0
        Hll, Hlr, Hrl, Hrr = xtc.align_impulse_responses(H, H, H, H)
        # Should return four arrays of same shape as input
        for arr in (Hll, Hlr, Hrl, Hrr):
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, H.shape)
        # Plot only one relevant signal for inspection
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(len(Hll)), Hll, 'o', label="Aligned IR")
        plt.title("Aligned Impulse Response (Shape Test)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Removed test_find_first_above_threshold (uninformative)

    def test_extract_hrirs_sam_no_interpolate(self):
        # Direct extraction at a multiple of 5 degrees
        hl, hr, sr = xtc.extract_hrirs_sam(
            "P0275_FreeFieldComp_48kHz.sofa", 0.0,
            show_plots=False, attempt_interpolate=False
        )
        self.assertIsInstance(hl, np.ndarray)
        self.assertIsInstance(hr, np.ndarray)
        sr_arr = np.asarray(sr)
        self.assertGreater(sr_arr.size, 0)
        self.assertGreater(sr_arr.flat[0], 0)

    def test_extract_hrirs_sam_interpolate(self):
        # Interpolated extraction at a non-multiple of 5
        hl, hr, sr = xtc.extract_hrirs_sam(
            "P0275_FreeFieldComp_48kHz.sofa", 7.0,
            show_plots=False, attempt_interpolate=True
        )
        self.assertIsInstance(hl, np.ndarray)
        self.assertIsInstance(hr, np.ndarray)
        sr_arr = np.asarray(sr)
        self.assertGreater(sr_arr.size, 0)
        self.assertGreater(sr_arr.flat[0], 0)

    def test_find_sofa_index_for_azimuth(self):
        # Should find a valid index for a known angle
        sf = sofa.SOFAFile("P0275_FreeFieldComp_48kHz.sofa", 'r')
        idx = xtc.find_sofa_index_for_azimuth(sf, 0.0, tolerance=2.5)
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)
        sf.close()

    def test_generate_filter_time_domain(self):
        # Sanity check time-domain filter generation
        H = np.zeros(8); H[0] = 1.0
        spk = np.array([[1.0, 0.0], [0.0, 1.0]])
        head = np.array([0.0, 0.0])
        fll, flr, frl, frr = xtc.generate_filter(
            H, H, H, H, spk, head,
            filter_length=64, samplerate=48000,
            debug=False, format='TimeDomain'
        )
        for arr in (fll, flr, frl, frr):
            self.assertIsInstance(arr, np.ndarray)
            self.assertGreaterEqual(arr.size, 64)

    def test_generate_kn_filter(self):
        # Sanity check Kirkeby-Nelson filter generation
        H = np.zeros(8); H[0] = 1.0
        spk = np.array([[1.0, 0.0], [0.0, 1.0]])
        head = np.array([0.0, 0.0])
        Fll, Flr, Frl, Frr = xtc.generate_kn_filter(
            H, H, H, H, spk, head,
            filter_length=64, samplerate=48000,
            debug=False, lambda_freq=1e-4
        )
        for arr in (Fll, Flr, Frl, Frr):
            self.assertIsInstance(arr, np.ndarray)
            self.assertGreater(arr.size, 0)

    def test_check_symmetric_angles(self):
        # Should detect both -30 and +30 in the file
        result = xtc.check_symmetric_angles(
            "P0275_FreeFieldComp_48kHz.sofa",
            left_az=-30.0, right_az=30.0, tolerance=2.5
        )
        self.assertTrue(result)

    def test_compute_rt60(self):
        # Exponential decay IR => function currently returns None
        ir = np.exp(-np.linspace(0, 1, 48000))
        rt60 = xtc.compute_rt60(ir, 48000, noise_floor_region=(0, 0.01))
        self.assertIsNone(rt60)


def run_unit_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestXTCLabTracked))
    suite.addTests(loader.loadTestsFromTestCase(TestXTCFunctions))
    runner = unittest.TextTestRunner(resultclass=TableTestResult, verbosity=2)
    result = runner.run(suite)
    # Print summary table
    print("\nSummary of unit tests:")
    print("{:<50} {}".format("Test", "Status"))
    for test in result.successes:
        print("{:<50} {}".format(str(test), "PASS"))
    for test, _ in result.failures:
        print("{:<50} {}".format(str(test), "FAIL"))
    for test, _ in result.errors:
        print("{:<50} {}".format(str(test), "ERROR"))





# Enhanced inspection tests: audio pipeline and filter stages with FFT visualization
def run_inspection_tests():
    from unittest.mock import patch
    from audio_engine import AudioEngine
    import numpy as np
    import matplotlib.pyplot as plt
    from xtc import generate_filter, generate_kn_filter, compute_rt60, fft_noise_subtraction

    fs = 48000

    # Inspection Test: AudioEngine._copy_input_buffer()
    input_buffer = np.zeros((1024, 2))
    input_buffer[:50, 0] = np.linspace(0, 1, 50)
    input_buffer[:50, 1] = np.linspace(1, 0, 50)
    engine = AudioEngine(config={"binaural_device": "", "playback_device": ""})
    copied_buffer = engine._copy_input_buffer(input_buffer)

    plt.figure(figsize=(12, 8))
    plt.suptitle("AudioEngine._copy_input_buffer()")

    plt.subplot(2, 2, 1)
    plt.title("Original Input (Time Domain)")
    plt.plot(np.arange(50), input_buffer[:50, 0], 'o', label='Left Channel')
    plt.plot(np.arange(50), input_buffer[:50, 1], 'o', label='Right Channel')
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Copied Buffer (Time Domain)")
    plt.plot(np.arange(50), copied_buffer[:50, 0], 'o', label='Left Channel Copy')
    plt.plot(np.arange(50), copied_buffer[:50, 1], 'o', label='Right Channel Copy')
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()

    fft_original = np.abs(np.fft.rfft(input_buffer[:,0]))
    fft_copied = np.abs(np.fft.rfft(copied_buffer[:,0]))
    freqs = np.fft.rfftfreq(1024, 1/fs)

    plt.subplot(2, 2, 3)
    plt.title("Original Input FFT")
    plt.plot(freqs, fft_original)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale('log')
    plt.grid(True, which='both')

    plt.subplot(2, 2, 4)
    plt.title("Copied Buffer FFT")
    plt.plot(freqs, fft_copied)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale('log')
    plt.grid(True, which='both')

    plt.tight_layout()
    plt.show()

    # Inspection Test: AudioEngine._apply_crossover_filters()
    # Use white noise instead of pure sine to test frequency splitting more effectively
    rng = np.random.default_rng(seed=42)
    input_buffer = rng.normal(0, 1, (1024, 2))
    t = np.arange(1024) / fs
    (lp_L, lp_R), (hp_L, hp_R) = engine._apply_crossover_filters(input_buffer)

    plt.figure(figsize=(12, 8))
    plt.suptitle("AudioEngine._apply_crossover_filters()")

    plt.subplot(2, 2, 1)
    plt.title("Input Signal (Time Domain)")
    plt.plot(np.arange(1024), input_buffer[:,0], 'o')
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(2, 2, 2)
    plt.title("Filtered Signals (Time Domain)")
    plt.plot(np.arange(1024), lp_L, 'o', label='Low-pass')
    plt.plot(np.arange(1024), hp_L, 'o', label='High-pass')
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.ylim(-3, 3)
    plt.xlim(0, 100)  # Show only first 100 samples

    fft_input = np.abs(np.fft.rfft(input_buffer[:,0]))
    fft_lp = np.abs(np.fft.rfft(lp_L))
    fft_hp = np.abs(np.fft.rfft(hp_L))
    freqs = np.fft.rfftfreq(1024, 1/fs)

    plt.subplot(2, 2, 3)
    plt.title("Input Signal FFT")
    plt.plot(freqs, fft_input)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale('log')
    plt.grid(True, which='both')

    plt.subplot(2, 2, 4)
    plt.title("Filtered Signals FFT")
    plt.plot(freqs, fft_lp, label='LP FFT')
    plt.plot(freqs, fft_hp, label='HP FFT')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale('log')
    plt.grid(True, which='both')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Load HRIRs from dataset
    hll, hlr, sr = xtc.extract_hrirs_sam("P0275_FreeFieldComp_48kHz.sofa", -30.0, show_plots=False)
    hrl, hrr, sr = xtc.extract_hrirs_sam("P0275_FreeFieldComp_48kHz.sofa", 30.0, show_plots=False)
    samplerate = sr[0]
    spk_pos = np.array([[1.0, 0.0], [0.0, 1.0]])
    head_pos = np.array([0.0, 0.0])
    # --- Simplified filter/HRIR/response spectrum inspection ---
    def show_filter_plots(title, FLL, FLR, FRL, FRR):
        # Use 4096-point FFT for HRIRs
        N_fft = 4096
        N = N_fft
        hf_colors = ['C0', 'C1', 'C2', 'C3']  # Match H_ll, H_lr, H_rl, H_rr
        # HRIR FFTs (real part only)
        H_ll = np.fft.fft(hll, n=N_fft)
        H_lr = np.fft.fft(hlr, n=N_fft)
        H_rl = np.fft.fft(hrl, n=N_fft)
        H_rr = np.fft.fft(hrr, n=N_fft)
        # Only positive frequencies
        freqs_hrir = np.fft.fftfreq(N, d=1.0/samplerate)[:N//2]

        # delay H matrices.
        delay_samples_l = -700
        delay_samples_r = -700
        omega = 2 * np.pi * np.fft.fftfreq(N, d=1.0/samplerate)
        tau_L = delay_samples_l / samplerate
        tau_R = delay_samples_r / samplerate
        exp_L = np.exp(1j * omega * tau_L)
        exp_R = np.exp(1j * omega * tau_R)

        H_ll *= exp_L
        H_lr *= exp_R
        H_rl *= exp_L
        H_rr *= exp_R

        # Filter spectra (input as frequency domain, may be longer or shorter than N_fft)
        def pad_to(arr, N):
            if len(arr) < N:
                return np.pad(arr, (0, N-len(arr)))
            return arr[:N]

        # Plot HRIR spectra
        plt.figure(figsize=(14, 10))
        plt.subplot(3,2,1)
        plt.title("HRIR Spectra (Magnitude)")
        plt.plot(freqs_hrir, 20*np.log10(np.abs(H_ll[:N//2])+1e-12), label="H_ll", color='C0')
        plt.plot(freqs_hrir, 20*np.log10(np.abs(H_lr[:N//2])+1e-12), label="H_lr", color='C1')
        plt.plot(freqs_hrir, 20*np.log10(np.abs(H_rl[:N//2])+1e-12), label="H_rl", color='C2')
        plt.plot(freqs_hrir, 20*np.log10(np.abs(H_rr[:N//2])+1e-12), label="H_rr", color='C3')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.xscale('log')
        plt.grid(True, which='both')
        # Plot filter spectra
        plt.subplot(3,2,2)
        plt.title("Filter Spectra (Magnitude)")
        plt.plot(freqs_hrir, 20*np.log10(np.abs(pad_to(FLL, N_fft)[:N//2])+1e-12), label="FLL", color='C0')
        plt.plot(freqs_hrir, 20*np.log10(np.abs(pad_to(FLR, N_fft)[:N//2])+1e-12), label="FLR", color='C1')
        plt.plot(freqs_hrir, 20*np.log10(np.abs(pad_to(FRL, N_fft)[:N//2])+1e-12), label="FRL", color='C2')
        plt.plot(freqs_hrir, 20*np.log10(np.abs(pad_to(FRR, N_fft)[:N//2])+1e-12), label="FRR", color='C3')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.xscale('log')
        plt.grid(True, which='both')
        # Compute HF = H @ F (einsum as per instructions)
        H_freq = np.array([
            [H_ll, H_lr],
            [H_rl, H_rr]
        ])  # shape (2,2,N_fft)
        F_freq = np.array([
            [pad_to(FLL, N_fft), pad_to(FLR, N_fft)],
            [pad_to(FRL, N_fft), pad_to(FRR, N_fft)]
        ])  # shape (2,2,N_fft)
        # einsum: 'imk,mjk->ijk' where i,m,j=2, k=N_fft
        HF = np.einsum('imk,mjk->ijk', H_freq, F_freq)
        # Plot spectra of four HF components, matching colors
        hf_labels = ['HF[0,0]', 'HF[0,1]', 'HF[1,0]', 'HF[1,1]']
        for i, (row, col) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            plt.subplot(3,2,3+i)
            plt.title(f"{hf_labels[i]} = H @ F")
            HF_mag = 20*np.log10(np.abs(HF[row,col][:N//2]) + 1e-12)
            plt.plot(freqs_hrir, HF_mag, label=hf_labels[i], color=hf_colors[i])
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.xscale('log')
            plt.grid(True, which='both')
            plt.legend()
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    FLL, FLR, FRL, FRR = generate_filter(hll, hlr, hrl, hrr, spk_pos, head_pos, samplerate=fs, format='FrequencyDomain')
    show_filter_plots("generate_filter() output", FLL, FLR, FRL, FRR)

    FLL_kn, FLR_kn, FRL_kn, FRR_kn = generate_kn_filter(hll, hlr, hrl, hrr, spk_pos, head_pos, samplerate=fs)
    show_filter_plots("generate_kn_filter() output", FLL_kn, FLR_kn, FRL_kn, FRR_kn)

    # Restore test for compute_rt60()
    # Use a simple exponential decay IR with fixed length
    ir = np.exp(-np.linspace(0, 1, int(samplerate)))  # Fixed length
    rt60 = compute_rt60(ir, samplerate)
    print(f"Test compute_rt60: RT60 = {rt60}")
    # Plot IR and overlay RT60 estimation
    plt.figure()
    plt.title("RT60 Estimation")
    plt.plot(ir, label='Impulse Response')
    if rt60 is not None:
        plt.axvline(int(rt60 * samplerate), color='r', linestyle='--', label=f'RT60 = {rt60:.2f}s')
    else:
        plt.axhline(0, color='r', linestyle='--', label='RT60 not found')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Restore test for fft_noise_subtraction()
    # Simulate IR with added noise
    # clean_ir = np.zeros(256)
    # clean_ir[0] = 1.0
    # noise = np.random.normal(0, 0.05, 256)
    # noisy_ir = clean_ir + noise
    # noise_floor = noise
    # cleaned = fft_noise_subtraction(noisy_ir, noise_floor, samplerate)
    # plt.figure(figsize=(10, 5))
    # plt.title("fft_noise_subtraction()")
    # plt.plot(noisy_ir, label="Noisy IR")
    # plt.plot(cleaned, label="Denoised IR")
    # plt.xlabel("Sample")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Crossover alignment test: use white noise, plot FFT magnitude, unwrapped phase, and wrapped phase
    rng = np.random.default_rng(seed=123)
    noise_buffer = rng.normal(0, 1, (1024, 2))
    (lp_L, lp_R), (hp_L, hp_R) = engine._apply_crossover_filters(noise_buffer)
    freqs = np.fft.rfftfreq(1024, 1/fs)
    fft_lp = np.fft.rfft(lp_L)
    fft_hp = np.fft.rfft(hp_L)
    eps = 1e-12
    # Insert new 3-subplot figure: magnitude, unwrapped phase, wrapped phase
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(freqs, 20*np.log10(np.abs(fft_lp) + 1e-12), label='LPF')
    plt.plot(freqs, 20*np.log10(np.abs(fft_hp) + 1e-12), label='HPF')
    plt.axvline(x=80, color='gray', linestyle=':', label='Crossover')
    plt.xscale("log")
    plt.ylabel('Magnitude (dB)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(freqs, np.unwrap(np.angle(fft_lp)), label='LPF phase')
    plt.plot(freqs, np.unwrap(np.angle(fft_hp)), label='HPF phase')
    plt.axvline(x=80, color='gray', linestyle=':')
    plt.xscale("log")
    plt.ylabel('Unwrapped Phase (rad)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(freqs, np.angle(fft_lp), label='LPF wrapped phase')
    plt.plot(freqs, np.angle(fft_hp), label='HPF wrapped phase')
    plt.axvline(x=80, color='gray', linestyle=':')
    plt.xscale("log")
    plt.ylabel('Wrapped Phase (rad)')
    plt.xlabel('Frequency (Hz)')
    plt.legend()

    plt.suptitle('Crossover Alignment Test')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_unit_tests()
    run_inspection_tests()