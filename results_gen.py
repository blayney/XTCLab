import numpy as np
import matplotlib.pyplot as plt
import xtc
import spatialaudiometrics.load_data
import pysofaconventions as sofa
from scipy.interpolate import make_interp_spline

def plot_metric_curve(x_vals, y_vals, ylabel, title, out_path, invert=False):
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    x = np.array(x_vals)
    y = np.array(y_vals)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y, k=3)(x_smooth)

    # Use a fixed dark color for the line
    line_color = 'purple' if invert else 'navy'

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_smooth, y_smooth, color=line_color, linewidth=2)

    ax.set_xlabel("Span (degrees)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
def generate_rotational_results(filter_function, outdir, label):
    # Load SOFA file and extract HRTF data
    hrtf = spatialaudiometrics.load_data.HRTF("/Users/willblayney/Development/FYP/XTCLab/P0275_FreeFieldComp_48kHz.sofa")

    # Filter azimuths at elevation = 0
    azimuths = sorted(set(hrtf.locs[:,0][hrtf.locs[:,1] == 0]))
    hrirs_left = []
    hrirs_right = []
    valid_azimuths = []

    for az in azimuths:
        idx = np.where((hrtf.locs[:,0] == az) & (hrtf.locs[:,1] == 0))[0]
        if len(idx) > 0:
            idx = idx[0]
            hrirs_left.append(hrtf.hrir[idx, 0, :])
            hrirs_right.append(hrtf.hrir[idx, 1, :])
            valid_azimuths.append(az)

    hrirs_left = np.array(hrirs_left)
    hrirs_right = np.array(hrirs_right)
    valid_azimuths = np.array(valid_azimuths)
    samplerate = sofa.SOFAFile("/Users/willblayney/Development/FYP/XTCLab/P0275_FreeFieldComp_48kHz.sofa", 'r').getSamplingRate()

    # Speaker positions
    speaker_positions = xtc.compute_speaker_positions_2D(1.5, -30, 30)
    head_position = np.array([0.0, 0.0])

    # Initialize results
    rotational_results = {}
    filter_matrices = {}

    for az in range(-90, 91, 5):
        # For rotational test, head rotates, speaker positions fixed
        # Head at azimuth = az
        # Compute HRIRs for speakers at -30 and +30 deg relative to head
        left_az = (az - 30) % 360
        right_az = (az + 30) % 360

        try:
            idx_L = np.where((hrtf.locs[:,0] == left_az) & (hrtf.locs[:,1] == 0))[0][0]
            idx_R = np.where((hrtf.locs[:,0] == right_az) & (hrtf.locs[:,1] == 0))[0][0]
        except IndexError:
            print(f"[{label}] Skipping azimuth {az}: missing HRIRs")
            continue

        H_LL = hrtf.hrir[idx_L, 0, :]
        H_LR = hrtf.hrir[idx_L, 1, :]
        H_RL = hrtf.hrir[idx_R, 0, :]
        H_RR = hrtf.hrir[idx_R, 1, :]

        F_LL, F_LR, F_RL, F_RR = filter_function(H_LL, H_LR, H_RL, H_RR,
                                                 speaker_positions, head_position,
                                                 samplerate=samplerate, format='FrequencyDomain')

        # --- Plot and save filter FFT magnitude spectra at azimuth = 0 ---
        if az == 0:
            import os
            N = len(F_LL)
            freqs = np.fft.rfftfreq(N, d=1/samplerate)
            plt.figure(figsize=(12, 6))
            plt.plot(freqs, 20*np.log10(np.abs(F_LL[:len(freqs)])+1e-12), label='F_LL', color='blue')
            plt.plot(freqs, 20*np.log10(np.abs(F_LR[:len(freqs)])+1e-12), label='F_LR', color='green')
            plt.plot(freqs, 20*np.log10(np.abs(F_RL[:len(freqs)])+1e-12), label='F_RL', color='purple')
            plt.plot(freqs, 20*np.log10(np.abs(F_RR[:len(freqs)])+1e-12), label='F_RR', color='orange')
            plt.title(f"Loudspeaker Filter Curves ({label})")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            os.makedirs(os.path.join(outdir, "rotational"), exist_ok=True)
            plt.savefig(os.path.join(outdir, "rotational", "filter_ffts.png"))
            plt.close()
        # --- end FFT plot block ---

        H_ll = np.fft.fft(H_LL, n=len(F_LL))
        H_lr = np.fft.fft(H_LR, n=len(F_LL))
        H_rl = np.fft.fft(H_RL, n=len(F_LL))
        H_rr = np.fft.fft(H_RR, n=len(F_LL))

        H_freq = np.array([[H_ll, H_lr], [H_rl, H_rr]])
        F_freq = np.array([[F_LL, F_LR], [F_RL, F_RR]])

        Y_freq = np.zeros((2, 2, len(F_LL)), dtype=complex)
        for i in range(2):
            for j in range(2):
                Y_freq[i, j, :] = F_freq[i, 0] * H_freq[0, j] + F_freq[i, 1] * H_freq[1, j]

        Y_LL = Y_freq[0, 0, :]
        Y_LR = Y_freq[0, 1, :]
        Y_RR = Y_freq[1, 1, :]
        Y_RL = Y_freq[1, 0, :]

        eps = 1e-12
        D_left = 20*np.log10(np.abs(Y_LL) + eps) - 20*np.log10(np.abs(Y_LR) + eps)
        D_right = 20*np.log10(np.abs(Y_RR) + eps) - 20*np.log10(np.abs(Y_RL) + eps)
        D_avg = (D_left + D_right) / 2.0

        mu_D = np.mean(D_avg)
        sigma_D = np.std(D_avg, ddof=1)

        T_ipsi = 20*np.log10(np.abs(Y_LL) + eps)
        mu_T_ipsi = np.mean(T_ipsi)
        sigma_T_ipsi = np.std(T_ipsi, ddof=1)
        sigma_col = np.sqrt(np.sum(T_ipsi**2) / (len(T_ipsi) - 1))

        C_val = mu_D / (1.0 + sigma_T_ipsi)

        rotational_results[az] = {
            'mu_D': mu_D,
            'sigma_D': sigma_D,
            'sigma_T_ipsi': sigma_T_ipsi,
            'sigma_col': sigma_col,
            'C': C_val
        }

        filter_matrices[az] = F_freq

    import os
    rotational_dir = f"{outdir}/rotational"
    os.makedirs(rotational_dir, exist_ok=True)

    # Extract azimuths and metrics
    azimuths = sorted(rotational_results.keys())
    mu_D_vals = [rotational_results[a]['mu_D'] for a in azimuths]
    sigma_D_vals = [rotational_results[a]['sigma_D'] for a in azimuths]
    sigma_T_vals = [rotational_results[a]['sigma_T_ipsi'] for a in azimuths]
    sigma_col_vals = [rotational_results[a]['sigma_col'] for a in azimuths]
    C_vals = [rotational_results[a]['C'] for a in azimuths]

    # Plot performance metrics
    plot_metric_curve(azimuths, mu_D_vals, 'μ_D', f'μ_D vs Azimuth ({label})', f"{rotational_dir}/mu_D_vs_azimuth.png", invert=False)
    plot_metric_curve(azimuths, sigma_D_vals, 'σ_D', f'σ_D vs Azimuth ({label})', f"{rotational_dir}/sigma_D_vs_azimuth.png", invert=True)
    plot_metric_curve(azimuths, sigma_T_vals, 'σ_T_ipsi', f'σ_T_ipsi vs Azimuth ({label})', f"{rotational_dir}/sigma_T_ipsi_vs_azimuth.png", invert=True)
    plot_metric_curve(azimuths, sigma_col_vals, 'σ_col', f'σ_col vs Azimuth ({label})', f"{rotational_dir}/sigma_col_vs_azimuth.png", invert=True)
    plot_metric_curve(azimuths, C_vals, 'C', f'C Index vs Azimuth ({label})', f"{rotational_dir}/C_vs_azimuth.png", invert=False)
    
    print(f"[{label}] Rotational results:")
    for k, v in rotational_results.items():
        print(f"[{label}] Azimuth {k}°: {v}")

    # Plot filters and TFs at specific azimuths
    for az in [-60, 0, 60]:
        if az not in filter_matrices:
            continue
        F_freq = filter_matrices[az]
        labels = ['F_LL', 'F_LR', 'F_RL', 'F_RR']
        colors = ['b', 'g', 'r', 'm']
        f_time = [np.fft.ifft(F_freq[i, j]).real for i in range(2) for j in range(2)]
        t = np.arange(len(f_time[0])) / samplerate

        # Plot impulse responses in 2×2 matrix layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i in range(2):
            for j in range(2):
                k = i * 2 + j
                axes[i, j].plot(t, f_time[k], color=colors[k])
                axes[i, j].set_title(labels[k])
                axes[i, j].set_xlabel("Time (s)")
                axes[i, j].set_ylabel("Amplitude")
                axes[i, j].grid(True)
        plt.suptitle(f"Filter Impulse Responses (Azimuth={az}°, {label})")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{rotational_dir}/impulse_responses_azimuth_{az}.png")
        plt.close()

        # Get forward H matrix at az
        left_az = (az - 30) % 360
        right_az = (az + 30) % 360
        try:
            idx_L = np.where((hrtf.locs[:,0] == left_az) & (hrtf.locs[:,1] == 0))[0][0]
            idx_R = np.where((hrtf.locs[:,0] == right_az) & (hrtf.locs[:,1] == 0))[0][0]
        except IndexError:
            continue

        H_LL = hrtf.hrir[idx_L, 0, :]
        H_LR = hrtf.hrir[idx_L, 1, :]
        H_RL = hrtf.hrir[idx_R, 0, :]
        H_RR = hrtf.hrir[idx_R, 1, :]

        N = len(F_freq[0,0])
        H_freq = np.array([
            [np.fft.fft(H_LL, n=N), np.fft.fft(H_LR, n=N)],
            [np.fft.fft(H_RL, n=N), np.fft.fft(H_RR, n=N)]
        ])
        Y_freq = np.zeros((2, 2, N), dtype=complex)
        for i in range(2):
            for j in range(2):
                Y_freq[i, j, :] = F_freq[i, 0] * H_freq[0, j] + F_freq[i, 1] * H_freq[1, j]

        freqs = np.fft.fftfreq(N, d=1/samplerate)
        pos_freqs = freqs[:N//2]
        Y_LL = Y_freq[0, 0, :N//2]
        Y_LR = Y_freq[0, 1, :N//2]
        Y_RR = Y_freq[1, 1, :N//2]
        Y_RL = Y_freq[1, 0, :N//2]

        # Plot TF magnitude responses
        # Left ear TF
        plt.figure(figsize=(10, 4))
        plt.plot(pos_freqs, 20*np.log10(np.abs(Y_LL)+1e-12), label='Y_LL (ipsi)', color='blue')
        plt.plot(pos_freqs, 20*np.log10(np.abs(Y_LR)+1e-12), label='Y_LR (contra)', linestyle='--', color='red')
        plt.title(f"Left Ear TF (Azimuth={az}°, {label})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{rotational_dir}/transfer_functions_left_azimuth_{az}.png")
        plt.close()

        # Right ear TF
        plt.figure(figsize=(10, 4))
        plt.plot(pos_freqs, 20*np.log10(np.abs(Y_RR)+1e-12), label='Y_RR (ipsi)', color='blue')
        plt.plot(pos_freqs, 20*np.log10(np.abs(Y_RL)+1e-12), label='Y_RL (contra)', linestyle='--', color='red')
        plt.title(f"Right Ear TF (Azimuth={az}°, {label})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{rotational_dir}/transfer_functions_right_azimuth_{az}.png")
        plt.close()


# New static test: head at 0°, sweep speaker span (±span), symmetric layout
def generate_static_results(filter_function, outdir, label):
    # Load SOFA file and extract HRTF data
    hrtf = spatialaudiometrics.load_data.HRTF("/Users/willblayney/Development/FYP/XTCLab/P0275_FreeFieldComp_48kHz.sofa")
    samplerate = sofa.SOFAFile("/Users/willblayney/Development/FYP/XTCLab/P0275_FreeFieldComp_48kHz.sofa", 'r').getSamplingRate()

    # Head is always at 0°
    head_position = np.array([0.0, 0.0])

    static_results = {}
    filter_matrices = {}

    # Sweep spans from 5 to 90 in steps of 5 (degrees)
    spans = list(range(5, 91, 5))
    for span in spans:
        left_az = (-span) % 360
        right_az = (span) % 360
        # Speakers at ±span at 1.5m
        speaker_positions = xtc.compute_speaker_positions_2D(1.5, -span, span)
        try:
            idx_L = np.where((hrtf.locs[:,0] == left_az) & (hrtf.locs[:,1] == 0))[0][0]
            idx_R = np.where((hrtf.locs[:,0] == right_az) & (hrtf.locs[:,1] == 0))[0][0]
        except IndexError:
            print(f"[{label}] Skipping span {span}: missing HRIRs")
            continue

        H_LL = hrtf.hrir[idx_L, 0, :]
        H_LR = hrtf.hrir[idx_L, 1, :]
        H_RL = hrtf.hrir[idx_R, 0, :]
        H_RR = hrtf.hrir[idx_R, 1, :]

        F_LL, F_LR, F_RL, F_RR = filter_function(H_LL, H_LR, H_RL, H_RR,
                                                 speaker_positions, head_position,
                                                 samplerate=samplerate, format='FrequencyDomain')

        # Plot and save filter FFT magnitude spectra at 30 deg span
        if span == 30:
            import os
            N = len(F_LL)
            freqs = np.fft.rfftfreq(N, d=1/samplerate)
            plt.figure(figsize=(12, 6))
            plt.plot(freqs, 20*np.log10(np.abs(F_LL[:len(freqs)])+1e-12), label='F_LL', color='blue')
            plt.plot(freqs, 20*np.log10(np.abs(F_LR[:len(freqs)])+1e-12), label='F_LR', color='green')
            plt.plot(freqs, 20*np.log10(np.abs(F_RL[:len(freqs)])+1e-12), label='F_RL', color='purple')
            plt.plot(freqs, 20*np.log10(np.abs(F_RR[:len(freqs)])+1e-12), label='F_RR', color='orange')
            plt.title(f"Loudspeaker Filter Curves ({label})")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            os.makedirs(os.path.join(outdir, "static"), exist_ok=True)
            plt.savefig(os.path.join(outdir, "static", "filter_ffts.png"))
            plt.close()

        H_ll = np.fft.fft(H_LL, n=len(F_LL))
        H_lr = np.fft.fft(H_LR, n=len(F_LL))
        H_rl = np.fft.fft(H_RL, n=len(F_LL))
        H_rr = np.fft.fft(H_RR, n=len(F_LL))

        H_freq = np.array([[H_ll, H_lr], [H_rl, H_rr]])
        F_freq = np.array([[F_LL, F_LR], [F_RL, F_RR]])

        Y_freq = np.zeros((2, 2, len(F_LL)), dtype=complex)
        for i in range(2):
            for j in range(2):
                Y_freq[i, j, :] = F_freq[i, 0] * H_freq[0, j] + F_freq[i, 1] * H_freq[1, j]

        Y_LL = Y_freq[0, 0, :]
        Y_LR = Y_freq[0, 1, :]
        Y_RR = Y_freq[1, 1, :]
        Y_RL = Y_freq[1, 0, :]

        eps = 1e-12
        D_left = 20*np.log10(np.abs(Y_LL) + eps) - 20*np.log10(np.abs(Y_LR) + eps)
        D_right = 20*np.log10(np.abs(Y_RR) + eps) - 20*np.log10(np.abs(Y_RL) + eps)
        D_avg = (D_left + D_right) / 2.0

        mu_D = np.mean(D_avg)
        sigma_D = np.std(D_avg, ddof=1)

        T_ipsi = 20*np.log10(np.abs(Y_LL) + eps)
        mu_T_ipsi = np.mean(T_ipsi)
        sigma_T_ipsi = np.std(T_ipsi, ddof=1)
        sigma_col = np.sqrt(np.sum(T_ipsi**2) / (len(T_ipsi) - 1))

        C_val = mu_D / (1.0 + sigma_T_ipsi)

        static_results[span] = {
            'mu_D': mu_D,
            'sigma_D': sigma_D,
            'sigma_T_ipsi': sigma_T_ipsi,
            'sigma_col': sigma_col,
            'C': C_val
        }

        filter_matrices[span] = F_freq

    import os
    static_dir = f"{outdir}/static"
    os.makedirs(static_dir, exist_ok=True)

    # Extract spans and metrics
    spans = sorted(static_results.keys())
    mu_D_vals = [static_results[s]['mu_D'] for s in spans]
    sigma_D_vals = [static_results[s]['sigma_D'] for s in spans]
    sigma_T_vals = [static_results[s]['sigma_T_ipsi'] for s in spans]
    sigma_col_vals = [static_results[s]['sigma_col'] for s in spans]
    C_vals = [static_results[s]['C'] for s in spans]

    # Plot performance metrics
    plot_metric_curve(spans, mu_D_vals, 'μ_D', f'μ_D vs Span ({label})', f"{static_dir}/mu_D_vs_span.png", invert=False)
    plot_metric_curve(spans, sigma_D_vals, 'σ_D', f'σ_D vs Span ({label})', f"{static_dir}/sigma_D_vs_span.png", invert=True)
    plot_metric_curve(spans, sigma_T_vals, 'σ_T_ipsi', f'σ_T_ipsi vs Span ({label})', f"{static_dir}/sigma_T_ipsi_vs_span.png", invert=True)
    plot_metric_curve(spans, sigma_col_vals, 'σ_col', f'σ_col vs Span ({label})', f"{static_dir}/sigma_col_vs_span.png", invert=True)
    plot_metric_curve(spans, C_vals, 'C', f'C Index vs Span ({label})', f"{static_dir}/C_vs_span.png", invert=False)
    
    print(f"[{label}] Static results:")
    for k, v in static_results.items():
        print(f"[{label}] Span {k}°: {v}")

    # Plot filters and TFs at specific spans
    for span in [10, 30, 90]:
        if span not in filter_matrices:
            continue
        F_freq = filter_matrices[span]
        labels = ['F_LL', 'F_LR', 'F_RL', 'F_RR']
        colors = ['b', 'g', 'r', 'm']
        f_time = [np.fft.ifft(F_freq[i, j]).real for i in range(2) for j in range(2)]
        t = np.arange(len(f_time[0])) / samplerate

        # Plot impulse responses in 2×2 matrix layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i in range(2):
            for j in range(2):
                k = i * 2 + j
                axes[i, j].plot(t, f_time[k], color=colors[k])
                axes[i, j].set_title(labels[k])
                axes[i, j].set_xlabel("Time (s)")
                axes[i, j].set_ylabel("Amplitude")
                axes[i, j].grid(True)
        plt.suptitle(f"Filter Impulse Responses (Span={span}°, {label})")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{static_dir}/impulse_responses_span_{span}.png")
        plt.close()

        # Get forward H matrix at span
        left_az = (-span) % 360
        right_az = (span) % 360
        try:
            idx_L = np.where((hrtf.locs[:,0] == left_az) & (hrtf.locs[:,1] == 0))[0][0]
            idx_R = np.where((hrtf.locs[:,0] == right_az) & (hrtf.locs[:,1] == 0))[0][0]
        except IndexError:
            continue

        H_LL = hrtf.hrir[idx_L, 0, :]
        H_LR = hrtf.hrir[idx_L, 1, :]
        H_RL = hrtf.hrir[idx_R, 0, :]
        H_RR = hrtf.hrir[idx_R, 1, :]

        N = len(F_freq[0,0])
        H_freq = np.array([
            [np.fft.fft(H_LL, n=N), np.fft.fft(H_LR, n=N)],
            [np.fft.fft(H_RL, n=N), np.fft.fft(H_RR, n=N)]
        ])
        Y_freq = np.zeros((2, 2, N), dtype=complex)
        for i in range(2):
            for j in range(2):
                Y_freq[i, j, :] = F_freq[i, 0] * H_freq[0, j] + F_freq[i, 1] * H_freq[1, j]

        freqs = np.fft.fftfreq(N, d=1/samplerate)
        pos_freqs = freqs[:N//2]
        Y_LL = Y_freq[0, 0, :N//2]
        Y_LR = Y_freq[0, 1, :N//2]
        Y_RR = Y_freq[1, 1, :N//2]
        Y_RL = Y_freq[1, 0, :N//2]

        # Plot TF magnitude responses
        # Left ear TF
        plt.figure(figsize=(10, 4))
        plt.plot(pos_freqs, 20*np.log10(np.abs(Y_LL)+1e-12), label='Y_LL (ipsi)', color='blue')
        plt.plot(pos_freqs, 20*np.log10(np.abs(Y_LR)+1e-12), label='Y_LR (contra)', linestyle='--', color='red')
        plt.title(f"Left Ear TF (Span={span}°, {label})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{static_dir}/transfer_functions_left_span_{span}.png")
        plt.close()

        # Right ear TF
        plt.figure(figsize=(10, 4))
        plt.plot(pos_freqs, 20*np.log10(np.abs(Y_RR)+1e-12), label='Y_RR (ipsi)', color='blue')
        plt.plot(pos_freqs, 20*np.log10(np.abs(Y_RL)+1e-12), label='Y_RL (contra)', linestyle='--', color='red')
        plt.title(f"Right Ear TF (Span={span}°, {label})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{static_dir}/transfer_functions_right_span_{span}.png")
        plt.close()


# Unified function for transfer function surfaces (static & rotational)
def plot_transfer_function_surfaces(filter_function, outdir, label, mode):
    import os
    from mpl_toolkits.mplot3d import Axes3D

    hrtf = spatialaudiometrics.load_data.HRTF("/Users/willblayney/Development/FYP/XTCLab/P0275_FreeFieldComp_48kHz.sofa")
    samplerate = sofa.SOFAFile("/Users/willblayney/Development/FYP/XTCLab/P0275_FreeFieldComp_48kHz.sofa", 'r').getSamplingRate()
    head_position = np.array([0.0, 0.0])

    if mode == 'static':
        # Head fixed at 0°, speakers sweep from 0 to 180° (±span)
        az_vals = list(range(0, 181, 5))
        outdir_mode = os.path.join(outdir, "static")
        os.makedirs(outdir_mode, exist_ok=True)
        span_grid_label = "Span (°)"
    elif mode == 'rotational':
        # Speakers fixed at ±30°, head azimuth sweeps from 0 to 180°
        az_vals = list(range(0, 181, 5))
        outdir_mode = os.path.join(outdir, "rotational")
        os.makedirs(outdir_mode, exist_ok=True)
        span_grid_label = "Head Azimuth (°)"
    else:
        raise ValueError("Mode must be 'static' or 'rotational'")

    freqs = None
    ipsi_L, contra_L, ipsi_R, contra_R = [], [], [], []
    actual_grid_vals = []

    for az in az_vals:
        if mode == 'static':
            # Speakers at ±span, head at 0°
            span = az
            left_az = (-span) % 360
            right_az = (span) % 360
            speaker_positions = xtc.compute_speaker_positions_2D(1.5, -span, span)
        elif mode == 'rotational':
            # Head at az, speakers fixed at ±30°
            head_az = az
            left_az = (head_az - 30) % 360
            right_az = (head_az + 30) % 360
            speaker_positions = xtc.compute_speaker_positions_2D(1.5, -30, 30)
        try:
            idx_L = np.where((hrtf.locs[:,0] == left_az) & (hrtf.locs[:,1] == 0))[0][0]
            idx_R = np.where((hrtf.locs[:,0] == right_az) & (hrtf.locs[:,1] == 0))[0][0]
        except IndexError:
            continue

        H_LL = hrtf.hrir[idx_L, 0, :]
        H_LR = hrtf.hrir[idx_L, 1, :]
        H_RL = hrtf.hrir[idx_R, 0, :]
        H_RR = hrtf.hrir[idx_R, 1, :]

        F_LL, F_LR, F_RL, F_RR = filter_function(H_LL, H_LR, H_RL, H_RR,
                                                 speaker_positions, head_position,
                                                 samplerate=samplerate, format='FrequencyDomain')

        N = len(F_LL)
        H_freq = np.array([
            [np.fft.fft(H_LL, n=N), np.fft.fft(H_LR, n=N)],
            [np.fft.fft(H_RL, n=N), np.fft.fft(H_RR, n=N)]
        ])
        F_freq = np.array([
            [F_LL, F_LR],
            [F_RL, F_RR]
        ])

        Y_freq = np.zeros((2, 2, N), dtype=complex)
        for i in range(2):
            for j in range(2):
                Y_freq[i, j, :] = F_freq[i, 0] * H_freq[0, j] + F_freq[i, 1] * H_freq[1, j]

        if freqs is None:
            freqs = np.fft.fftfreq(N, d=1/samplerate)[:N//2]

        Y_LL = Y_freq[0, 0, :N//2]
        Y_LR = Y_freq[0, 1, :N//2]
        Y_RR = Y_freq[1, 1, :N//2]
        Y_RL = Y_freq[1, 0, :N//2]

        ipsi_L.append(20*np.log10(np.abs(Y_LL)+1e-12))
        contra_L.append(20*np.log10(np.abs(Y_LR)+1e-12))
        ipsi_R.append(20*np.log10(np.abs(Y_RR)+1e-12))
        contra_R.append(20*np.log10(np.abs(Y_RL)+1e-12))
        actual_grid_vals.append(az)

    # Convert to arrays
    ipsi_L = np.array(ipsi_L)
    contra_L = np.array(contra_L)
    ipsi_R = np.array(ipsi_R)
    contra_R = np.array(contra_R)
    grid_vals = np.array(actual_grid_vals)
    span_grid, freq_grid = np.meshgrid(grid_vals, freqs, indexing='ij')

    def plot_surface(data1, data2, title1, title2, filename):
        fig = plt.figure(figsize=(16, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(freq_grid, span_grid, data1, color='navy', edgecolor='k', linewidth=0.1, antialiased=True)
        ax1.set_title(title1)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel(span_grid_label)
        ax1.set_zlabel("Magnitude (dB)")

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(freq_grid, span_grid, data2, color='purple', edgecolor='k', linewidth=0.1, antialiased=True)
        ax2.set_title(title2)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel(span_grid_label)
        ax2.set_zlabel("Magnitude (dB)")

        plt.tight_layout()
        plt.savefig(os.path.join(outdir_mode, filename))
        plt.close()

    plot_surface(ipsi_L, contra_L, "Left Ear Ipsilateral TF", "Left Ear Contralateral TF", "tf_surfaces_left.png")
    plot_surface(ipsi_R, contra_R, "Right Ear Ipsilateral TF", "Right Ear Contralateral TF", "tf_surfaces_right.png")

# Alias for adjugate filter (for symmetry)
def plot_transfer_function_surfaces_adjugate(*args, **kwargs):
    return plot_transfer_function_surfaces(*args, **kwargs)

# Call both static and rotational result generators for both filter types
generate_static_results(xtc.generate_filter, "results/default_filter", "Adjugate Filter")
generate_static_results(xtc.generate_kn_filter, "results/kn_filter", "KN Filter")
generate_rotational_results(xtc.generate_filter, "results/default_filter", "Adjugate Filter")
generate_rotational_results(xtc.generate_kn_filter, "results/kn_filter", "KN Filter")

# New unified transfer function surfaces for all combinations
plot_transfer_function_surfaces(xtc.generate_filter, "results/default_filter", "Adjugate Filter", "static")
plot_transfer_function_surfaces(xtc.generate_filter, "results/default_filter", "Adjugate Filter", "rotational")
plot_transfer_function_surfaces(xtc.generate_kn_filter, "results/kn_filter", "KN Filter", "static")
plot_transfer_function_surfaces(xtc.generate_kn_filter, "results/kn_filter", "KN Filter", "rotational")