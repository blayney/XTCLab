import numpy as np
import matplotlib.pyplot as plt
import xtc
import spatialaudiometrics.load_data
import pysofaconventions as sofa
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def loc_generate_filter(h_LL, h_LR, h_RL, h_RR, speaker_positions, head_position, head_angle=None, filter_length=2048, samplerate=48000, debug=False, format='TimeDomain'):
    L = max(len(h_LL), len(h_LR), len(h_RL), len(h_RR))
    # how many points? more points = more frequency domain precision at the expense of time domain precision.
    M = filter_length
    print("generate filter called")
    # To avoid aliasing, choose FFT length N = 2**ceil(log2(L_HRIR + filter_length - 1))
    N = 2**int(np.ceil(np.log2(L + filter_length - 1)))
    print("generate_filter: max length of H recordings:", L)
    print("requested filter length: ", filter_length)
    print("N: ", N)
    print("M: ", M)

    # compute H matrix in omega
    H_ll = np.fft.fft(h_LL, n=N)
    H_lr = np.fft.fft(h_LR, n=N)
    H_rl = np.fft.fft(h_RL, n=N)
    H_rr = np.fft.fft(h_RR, n=N)

    delay_samples_l = int(np.round(np.linalg.norm(speaker_positions[0] - head_position) / 343.0 * samplerate))
    delay_samples_r = int(np.round(np.linalg.norm(speaker_positions[1] - head_position) / 343.0 * samplerate))
    if delay_samples_l == delay_samples_r:
        #print("Left and right sources are coherent, no delay applied")
        tau_L = 0
        tau_R = 0
    elif delay_samples_l > delay_samples_r:
        print("right source arrives first, applying delay to right")
        tau_L = 0
        tau_R = (delay_samples_l - delay_samples_r)/samplerate
        print("applied {} seconds of delay to right".format(tau_R))
    elif delay_samples_r > delay_samples_l:
        print("left source arrives first, applying delay to left")
        tau_L = (delay_samples_r - delay_samples_l)/samplerate
        tau_R = 0
        print("applied {} seconds of delay to left".format(tau_L))
    
    if delay_samples_l != 0 or delay_samples_r != 0:
        freqs = np.fft.fftfreq(N, d=1/samplerate)
        omega = 2 * np.pi * freqs
    
        exp_L = np.exp(1j * omega * tau_L)    # for the left speaker
        exp_R = np.exp(1j * omega * tau_R)    # for the right speaker

        H_ll = H_ll * exp_L
        H_lr = H_lr * exp_R
        H_rl = H_rl * exp_L
        H_rr = H_rr * exp_R


    # Closed-form 2×2 inversion per bin (small ε to avoid division by zero)
    eps = 1e-12
    det = H_ll * H_rr - H_lr * H_rl + eps

    FLL =  H_rr / det
    FLR = -H_lr / det
    FRL = -H_rl / det
    FRR =  H_ll / det

    if debug & False:
        # Plot magnitude spectra of HRTF, filter, HF product, and filter impulse responses
        freqs = np.fft.fftfreq(N, d=1.0/samplerate)
        positive = freqs[:N//2]
        band = (positive >= 20) & (positive <= 20000)
        
        # Prepare H and F spectra lists
        specs_H = [H_ll, H_lr, H_rl, H_rr]
        specs_F = [F11, F12, F21, F22]
        labels = ['11','12','21','22']

        # Compute HF = H @ F per bin
        H_freq = np.array([[H_ll, H_lr], [H_rl, H_rr]])
        F_freq = np.array([[FLL, FLR], [FRL, FRR]])
        HF = np.einsum('imk,mjk->ijk', H_freq, F_freq)
        specs_HF = [HF[0,0,:], HF[0,1,:], HF[1,0,:], HF[1,1,:]]

        # Compute raw time-domain IRs (unwindowed)
        f_time = [np.fft.ifft(spec, n=N).real for spec in specs_F]
        # time axis for full IRs
        t_full = np.arange(N) / samplerate

        # Create 4×4 grid
        fig, axes = plt.subplots(4, 4, figsize=(18, 10))
        for i in range(4):
            # HRTF magnitude
            axes[i,0].plot(positive[band], 20*np.log10(np.abs(specs_H[i][:N//2][band]) + 1e-12),
                           label=f'H_{{{labels[i]}}}')
            axes[i,0].set_ylabel('Mag (dB)')
            axes[i,0].legend()
            # Filter magnitude
            axes[i,1].plot(positive[band], 20*np.log10(np.abs(specs_F[i][:N//2][band]) + 1e-12),
                           label=f'F_{{{labels[i]}}}', linestyle='--')
            axes[i,1].set_ylabel('Mag (dB)')
            axes[i,1].legend()
            # HF product magnitude
            mag_HF = 20*np.log10(np.clip(np.abs(specs_HF[i][:N//2]), 1e-16, None))
            axes[i,2].plot(positive, mag_HF, label=f"HF_{labels[i]}")
            axes[i,2].set_ylabel('Mag (dB)')
            axes[i,2].legend()
            # Time-domain impulse response (full IR)
            axes[i,3].plot(t_full, f_time[i], label=f'f_{{{labels[i]}}}(t)')
            axes[i,3].set_ylabel('Amp')
            axes[i,3].legend()
        # Label bottom axes
        for j, ax in enumerate(axes[3,:]):
            ax.set_xlabel('Frequency (Hz)' if j < 3 else 'Time (s)')
        fig.suptitle('HRTF vs. Filter vs. HF Product vs. Filter IRs')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    # Delay compensation
    freqs = np.fft.fftfreq(N, d=1/samplerate)
    omega = 2 * np.pi * freqs
    # delay_samples_l = int(np.round((np.linalg.norm(speaker_positions[0] - head_position) / 343.0) * samplerate))
    # delay_samples_r = int(np.round((np.linalg.norm(speaker_positions[1] - head_position) / 343.0) * samplerate))
    # I was previously rolling by the delay, but this is not correct.
    # Now I'm rolling by a static delay of 200 samples to prevent non-causality.
    delay_samples_l = -700
    delay_samples_r = -700

    tau_L = delay_samples_l / samplerate
    tau_R = delay_samples_r / samplerate
    exp_L = np.exp(1j * omega * tau_L)    # for the left speaker
    exp_R = np.exp(1j * omega * tau_R)    # for the right speaker

    FLL *= exp_L
    FLR *= exp_R
    FRL *= exp_L
    FRR *= exp_R

    HLL_comp = H_ll * exp_L
    HLR_comp = H_lr * exp_R
    HRL_comp = H_rl * exp_L
    HRR_comp = H_rr * exp_R

    fll = np.fft.ifft(FLL, n=N).real
    flr = np.fft.ifft(FLR, n=N).real
    frl = np.fft.ifft(FRL, n=N).real
    frr = np.fft.ifft(FRR, n=N).real

    # Step 4: window IRs to half filter_length, then roll delays
    def window_and_trim(ir, delay_samples=None):
        # 1) Truncate or pad ir to exactly filter_length
        if len(ir) < filter_length:
            ir2 = np.pad(ir, (0, filter_length - len(ir)), mode='constant')
        else:
            ir2 = ir[:filter_length]

        # 2) Build a full-length window, with a Hann bump of width half = filter_length//2
        win = np.zeros(filter_length)
        half = filter_length // 2
        hann = np.hanning(half)

        # 3) Compute where to start placing the Hann bump so its center is at delay_samples
        #    The Hann window's center index is half//2
        if delay_samples is None:
            start = 0
        else:
            start = delay_samples - (half // 2)

        # 4) Fill the bump into win, clipping to [0,filter_length)
        for i in range(half):
            idx = start + i
            if 0 <= idx < filter_length:
                win[idx] = hann[i]

        # 5) Apply
        return ir2 * win

    if debug:
        import matplotlib.pyplot as plt

        # Frequency axis (positive half)
        freqs = np.fft.fftfreq(N, d=1.0/samplerate)
        positive = freqs[:N//2]

        # Spectra of delayed H matrix (after delay compensation)
        specs_H_comp = [HLL_comp, HLR_comp, HRL_comp, HRR_comp]
        # Spectra of embedded-delay filters
        specs_F = [FLL, FLR, FRL, FRR]

        # Compute HF = H_comp @ F per bin
        H_freq_comp = np.array([[HLL_comp, HLR_comp],
                                [HRL_comp, HRR_comp]])
        F_freq = np.array([[FLL, FLR],
                           [FRL, FRR]])
        HF = np.zeros((2, 2, N), dtype=np.complex128)
        for k in range(N):
            Hk = np.array([[HLL_comp[k], HLR_comp[k]],
                           [HRL_comp[k], HRR_comp[k]]])
            Fk = np.array([[FLL[k], FLR[k]],
                           [FRL[k], FRR[k]]])
            HF[:, :, k] = Hk @ Fk

        specs_HF = [HF[0,0,:], HF[0,1,:], HF[1,0,:], HF[1,1,:]]

        # Time-domain IRs
        f_time = [fll, flr, frl, frr]
        t = np.arange(len(fll)) / samplerate

        labels = ['LL','LR','RL','RR']
        fig, axes = plt.subplots(4, 4, figsize=(18, 10))
        for i in range(4):
            # H spectrum
            axes[i,0].plot(positive, 20*np.log10(np.abs(specs_H_comp[i][:N//2]) + 1e-12),
                           label=f'H_{labels[i]}')
            axes[i,0].set_ylabel('Mag (dB)')
            axes[i,0].legend()
            # F spectrum
            axes[i,1].plot(positive, 20*np.log10(np.abs(specs_F[i][:N//2]) + 1e-12),
                           label=f'F_{labels[i]}', linestyle='--')
            axes[i,1].set_ylabel('Mag (dB)')
            axes[i,1].legend()
            # HF spectrum
            axes[i,2].plot(positive, 20*np.log10(np.abs(specs_HF[i][:N//2]) + 1e-12),
                           label=f'HF_{labels[i]}', linestyle='-.')
            axes[i,2].set_ylabel('Mag (dB)')
            axes[i,2].legend()
            # Impulse response
            axes[i,3].plot(t, f_time[i], label=f'f_{labels[i]}(t)')
            axes[i,3].set_ylabel('Amp')
            axes[i,3].legend()
            # X-axis labels
            for col in range(4):
                if i == 3:
                    axes[i,col].set_xlabel('Freq (Hz)' if col < 3 else 'Time (s)')
        fig.suptitle('Delayed H, Embedded-Delay F, HF Product, and Filter IRs')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    if debug & False:
        plot_filter_surface(fll, samplerate, title="FLL Magnitude Surface")
        plot_filter_surface(flr, samplerate, title="FLR Magnitude Surface")
        plot_filter_surface(frl, samplerate, title="FRL Magnitude Surface")
        plot_filter_surface(frr, samplerate, title="FRR Magnitude Surface")
    
    print("Filter lengths: ", len(fll), len(flr), len(frl), len(frr))
    if format=='TimeDomain':
        # Return time-domain filters
        return fll, flr, frl, frr
    elif format=='FrequencyDomain':
        return FLL, FLR, FRL, FRR


import csv
import os
import numpy as np
from collections import defaultdict

# --- Memoization cache for generate_kn_filter ---
filter_cache = None
cache_stats = None
cache_stats_file = "filter_cache_stats.csv"
cache_tolerance = 0.2  # degrees

def init_filter_cache():
    global filter_cache, cache_stats
    from collections import defaultdict
    import os
    import csv

    filter_cache = {}
    cache_stats = defaultdict(int)

    if os.path.exists(cache_stats_file):
        os.remove(cache_stats_file)
    with open(cache_stats_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["TotalCalls", "CacheHits", "HitRate"])

def loc_generate_kn_filter(h_LL, h_LR, h_RL, h_RR, speaker_positions, head_position, head_angle=None, filter_length=2048, samplerate=48000, debug=False, lambda_freq=1e-4, format='FrequencyDomain'):

    from scipy.fft import fft, fftfreq
    c = 343.0
    H_mat = np.array([[h_LL, h_LR], [h_RL, h_RR]])

    N = 2**int(np.ceil(np.log2(len(H_mat[0][0]) + filter_length - 1)))
    omega = 2 * np.pi * fftfreq(N, d=1/samplerate)
    # Step 1: FFT of HRIRs
    H_f = np.zeros((2, 2, N), dtype=complex)

    for i in range(2):
        for j in range(2):
            H_f[i, j, :] = fft(H_mat[i][j], n=N)
    
    delay_samples_l = int(np.round(np.linalg.norm(speaker_positions[0] - head_position) / 343.0 * samplerate))
    delay_samples_r = int(np.round(np.linalg.norm(speaker_positions[1] - head_position) / 343.0 * samplerate))
    if delay_samples_l == delay_samples_r:
        # print("Left and right sources are coherent, no delay applied")
        tau_L = 0
        tau_R = 0
    elif delay_samples_l > delay_samples_r:
        print("right source arrives first, applying delay to right")
        tau_L = 0
        tau_R = (delay_samples_l - delay_samples_r)/samplerate
        print("applied {} seconds of delay to right".format(tau_R))
    elif delay_samples_r > delay_samples_l:
        print("left source arrives first, applying delay to left")
        tau_L = (delay_samples_r - delay_samples_l)/samplerate
        tau_R = 0
        print("applied {} seconds of delay to left".format(tau_L))
    
    if delay_samples_l != 0 or delay_samples_r != 0:
        freqs = np.fft.fftfreq(N, d=1/samplerate)
        omega = 2 * np.pi * freqs

        exp_L = np.exp(1j * omega * tau_L)    # for the left speaker
        exp_R = np.exp(1j * omega * tau_R)    # for the right speaker

        # Apply delays to the correct HRIR paths
        H_f[0, 0, :] *= exp_L  # H_LL
        H_f[0, 1, :] *= exp_R  # H_LR
        H_f[1, 0, :] *= exp_L  # H_RL
        H_f[1, 1, :] *= exp_R  # H_RR

    # Step 3: Frequency domain Kirkeby-Nelson inversion
    F_f = np.zeros((2, 2, N), dtype=complex)
    for k in range(N):
        Hk = H_f[:, :, k]
        Hk_H = Hk.conj().T
        lambda_k = lambda_freq if np.isscalar(lambda_freq) else lambda_freq(k, N)
        inv = np.linalg.inv(Hk_H @ Hk + lambda_k * np.eye(2)) @ Hk_H
        F_f[:, :, k] = inv  # targeting D = identity matrix

    FLL = F_f[0, 0, :]
    FLR = F_f[0, 1, :]
    FRL = F_f[1, 0, :]
    FRR = F_f[1, 1, :]
    # multiply by 700 sample delay.
    delay_samples = -700
    tau_L = delay_samples / samplerate
    tau_R = delay_samples / samplerate
    exp_L = np.exp(1j * omega * tau_L)    # for the left speaker
    exp_R = np.exp(1j * omega * tau_R)    # for the right speaker
    FLL *= exp_L
    FLR *= exp_R
    FRL *= exp_L
    FRR *= exp_R

    return FLL, FLR, FRL, FRR  # fll, flr, frl, frr

def loc_generate_kn_filter_smart(
    h_LL, h_LR, h_RL, h_RR,
    speaker_positions, head_position, head_angle=None,
    filter_length=2048, samplerate=48000, debug=False, lambda_freq=1e-4, format='FrequencyDomain'
):
    from scipy.fft import fft, fftfreq
    c = 343.0
    H_mat = np.array([[h_LL, h_LR], [h_RL, h_RR]])

    N = 2 ** int(np.ceil(np.log2(len(H_mat[0][0]) + filter_length - 1)))
    omega = 2 * np.pi * fftfreq(N, d=1 / samplerate)
    # Step 1: FFT of HRIRs
    H_f = np.zeros((2, 2, N), dtype=complex)
    for i in range(2):
        for j in range(2):
            H_f[i, j, :] = fft(H_mat[i][j], n=N)

    delay_samples_l = int(np.round(np.linalg.norm(speaker_positions[0] - head_position) / c * samplerate))
    delay_samples_r = int(np.round(np.linalg.norm(speaker_positions[1] - head_position) / c * samplerate))
    if delay_samples_l == delay_samples_r:
        print("Left and right sources are coherent, no delay applied")
        tau_L = 0
        tau_R = 0
    elif delay_samples_l > delay_samples_r:
        print("right source arrives first, applying delay to right")
        tau_L = 0
        tau_R = (delay_samples_l - delay_samples_r) / samplerate
        print("applied {} seconds of delay to right".format(tau_R))
    elif delay_samples_r > delay_samples_l:
        print("left source arrives first, applying delay to left")
        tau_L = (delay_samples_r - delay_samples_l) / samplerate
        tau_R = 0
        print("applied {} seconds of delay to left".format(tau_L))

    if delay_samples_l != 0 or delay_samples_r != 0:
        freqs = np.fft.fftfreq(N, d=1 / samplerate)
        omega = 2 * np.pi * freqs
        exp_L = np.exp(1j * omega * tau_L)  # for the left speaker
        exp_R = np.exp(1j * omega * tau_R)  # for the right speaker
        # Apply delays to the correct HRIR paths
        H_f[0, 0, :] *= exp_L  # H_LL
        H_f[0, 1, :] *= exp_R  # H_LR
        H_f[1, 0, :] *= exp_L  # H_RL
        H_f[1, 1, :] *= exp_R  # H_RR

    # --- Helper functions for regularization optimization ---
    def compute_filter_response(H_f, lambda_w):
        N = H_f.shape[-1]
        F_f = np.zeros_like(H_f)
        for k in range(N):
            Hk = H_f[:, :, k]
            HkH = Hk @ Hk.conj().T
            try:
                inv_term = np.linalg.inv(HkH + lambda_w[k] * np.eye(2))
                F_f[:, :, k] = Hk.conj().T @ inv_term
            except np.linalg.LinAlgError:
                F_f[:, :, k] = np.zeros((2, 2), dtype=complex)
        return F_f
    
    def compute_regularization_error(log_lambdas, H_f, freqs, passband, beta=1.0):
        # Interpolate log-scale lambda across frequency, with safe handling for nonpositive freqs
        positive_freqs = freqs[freqs > 0]
        control_freqs = np.geomspace(positive_freqs[0], positive_freqs[-1], len(log_lambdas))
        lambda_interp_func = interp1d(np.log10(control_freqs), log_lambdas, kind='linear', fill_value="extrapolate")
        # Safe interpolation: only for positive frequencies
        log_lambda_w = np.full_like(freqs, fill_value=np.nan, dtype=float)
        safe_mask = freqs > 0
        freqs_slice = freqs[safe_mask]
        freqs_slice = freqs_slice.data
        log_lambda_w[safe_mask] = lambda_interp_func(np.log10(freqs_slice))
        lambda_w = 10 ** log_lambda_w
        # For negative/zero freqs, fallback to first positive value
        if np.any(safe_mask):
            lambda_w[~safe_mask] = lambda_w[safe_mask][0]
        else:
            lambda_w[:] = 1e-4  # fallback if no positive freqs (shouldn't happen)

        F_f = compute_filter_response(H_f, lambda_w)
        HF = np.einsum("ijk,jlk->ilk", H_f, F_f)  # H @ F

        # Extract diagonal and off-diagonal elements
        T_ipsi = np.abs(np.stack([HF[0, 0, :], HF[1, 1, :]]))
        T_contra = np.abs(np.stack([HF[0, 1, :], HF[1, 0, :]]))

        # Mask to passband only
        T_ipsi = T_ipsi[:, passband]
        T_contra = T_contra[:, passband]

        flatness_error = np.mean((20 * np.log10(T_ipsi + 1e-12)) ** 2)
        cancellation_error = np.mean(T_contra ** 2)

        if np.any(np.isnan(HF)):
            print("NaNs detected in HF matrix!")

        return flatness_error + beta * cancellation_error

    # Step 3: Frequency domain Kirkeby-Nelson inversion with optimized regularization profile
    freqs = np.fft.fftfreq(N, d=1 / samplerate)
    freq_mask = (freqs >= 100) & (freqs <= 7000)
    init_log_lambda = np.full(8, np.log10(1e-4))

    res = minimize(
        compute_regularization_error,
        init_log_lambda,
        args=(H_f, freqs, freq_mask),
        method='L-BFGS-B',
        bounds=[(-8, 2)] * 8,
        options={"maxiter": 50}
    )

    optimized_log_lambda = res.x
    positive_freqs = freqs[freqs > 0]
    control_freqs = np.geomspace(positive_freqs[0], positive_freqs[-1], len(optimized_log_lambda))
    lambda_interp_func = interp1d(np.log10(control_freqs), optimized_log_lambda, kind='linear', fill_value="extrapolate")
    # Safe interpolation: only for positive frequencies
    log_lambda_w = np.full_like(freqs, fill_value=np.nan, dtype=float)
    safe_mask = freqs > 0
    freqs_slice = freqs[safe_mask]
    freqs_slice = freqs_slice.data

    log_lambda_w[safe_mask] = lambda_interp_func(np.log10(freqs_slice))
    lambda_w = 10 ** log_lambda_w
    if np.any(safe_mask):
        lambda_w[~safe_mask] = lambda_w[safe_mask][0]
    else:
        lambda_w[:] = 1e-4

    F_f = compute_filter_response(H_f, lambda_w)

    FLL = F_f[0, 0, :]
    FLR = F_f[0, 1, :]
    FRL = F_f[1, 0, :]
    FRR = F_f[1, 1, :]
    # multiply by 700 sample delay.
    delay_samples = -700
    tau_L = delay_samples / samplerate
    tau_R = delay_samples / samplerate
    exp_L = np.exp(1j * omega * tau_L)    # for the left speaker
    exp_R = np.exp(1j * omega * tau_R)    # for the right speaker
    FLL *= exp_L
    FLR *= exp_R
    FRL *= exp_L
    FRR *= exp_R
    return FLL, FLR, FRL, FRR  # fll, flr, frl, frr

def plot_metric_curve(x_vals, y_vals, ylabel, title, out_path, invert=False):
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    x = np.array(x_vals)
    y = np.array(y_vals)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y, k=3)(x_smooth)

    # Update labels/titles for new filter naming conventions
    title = title.replace("kn filter smart", "VRRR Filter")
    title = title.replace("kn filter", "Ridge Regression Filter")
    title = title.replace("generate_filter", "Adjugate Filter")
    title = title.replace("default", "Adjugate Filter")
    ylabel = ylabel.replace("kn filter smart", "VRRR Filter")
    ylabel = ylabel.replace("kn filter", "RR Filter")
    ylabel = ylabel.replace("generate_filter", "Adjugate Filter")
    ylabel = ylabel.replace("default", "Adjugate Filter")

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
            # Update title and labels for new filter naming
            plot_title = f"Loudspeaker Filter Curves ({label})"
            plot_title = plot_title.replace("kn filter smart", "VRRR Filter")
            plot_title = plot_title.replace("kn filter", "Ridge Regression Filter")
            plot_title = plot_title.replace("generate_filter", "Adjugate Filter")
            plot_title = plot_title.replace("default", "Adjugate Filter")
            plt.title(plot_title)
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
        # ... (existing code for plotting, unchanged)

    # Save rotational_results to file for later comparison
    import json
    with open(os.path.join(rotational_dir, f"rotational_metrics_{label}.json"), "w") as f:
        json.dump(rotational_results, f, indent=2)

    # --- Save metrics.json for compatibility with load_rotational_metrics_json ---
    # Ensure outdir/rotational/ exists
    os.makedirs(os.path.join(outdir, "rotational"), exist_ok=True)
    metrics_json_path = os.path.join(outdir, "rotational", "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(rotational_results, f, indent=4)

    return rotational_results


# ------------------- COMPARISON SECTION -------------------

def compute_average_spectral_roughness(rotational_results):
    # Average std of time-domain IR (sigma_T_ipsi) over azimuths
    vals = [v['sigma_T_ipsi'] for v in rotational_results.values()]
    return float(np.mean(vals))

def compute_average_contralateral_reduction(rotational_results):
    # Average mu_D over azimuths
    vals = [v['mu_D'] for v in rotational_results.values()]
    return float(np.mean(vals))

def compute_average_ipsilateral_flatness(rotational_results):
    # Average 1/sigma_T_ipsi over azimuths
    vals = [v['sigma_T_ipsi'] for v in rotational_results.values()]
    vals = [v if v > 1e-8 else 1e-8 for v in vals]  # avoid div by 0
    inv_vals = [1.0/v for v in vals]
    return float(np.mean(inv_vals))

def plot_bar_comparison(values_dict, ylabel, title, out_path):
    import matplotlib.pyplot as plt
    names = list(values_dict.keys())
    vals = list(values_dict.values())
    fig, ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(names, vals, color=['teal','slateblue','darkorange'])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_C_vs_azimuth_comparison(azimuths, C_dict, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,5))
    for label, C_vals in C_dict.items():
        ax.plot(azimuths, C_vals, label=label)
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("C Index")
    ax.set_title("C Index vs Azimuth: Adjugate, RR, VRRR Filters")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def load_rotational_metrics_json(path):
    import json
    with open(path, "r") as f:
        data = json.load(f)
    # Convert keys to ints if needed
    results = {int(k): v for k, v in data.items()}
    return results

def run_filter_comparison():
    import os
    # Set up paths for previously saved rotational results
    base_dir = "results"
    comparison_dir = os.path.join(base_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    # Filenames (must match those saved by generate_rotational_results)
    adj_file = "results/default_filter/rotational/metrics.json"
    rr_file = "results/kn_filter/rotational/metrics.json"
    vrrr_file = "results/kn_filter_smart/rotational/metrics.json"
    # Load metrics
    adj = load_rotational_metrics_json(adj_file)
    rr = load_rotational_metrics_json(rr_file)
    vrrr = load_rotational_metrics_json(vrrr_file)

    # Compute average spectral roughness (std of IR)
    roughness = {
        "Adjugate Filter": compute_average_spectral_roughness(adj),
        "RR Filter": compute_average_spectral_roughness(rr),
        "VRRR Filter": compute_average_spectral_roughness(vrrr),
    }
    plot_bar_comparison(
        roughness,
        ylabel="Avg. Spectral Roughness (σ_T_ipsi)",
        title="Average Spectral Roughness Across Filters",
        out_path=os.path.join(comparison_dir, "spectral_roughness_bar.png")
    )

    # Compute average contralateral reduction (mean mu_D)
    contra = {
        "Adjugate Filter": compute_average_contralateral_reduction(adj),
        "RR Filter": compute_average_contralateral_reduction(rr),
        "VRRR Filter": compute_average_contralateral_reduction(vrrr),
    }
    plot_bar_comparison(
        contra,
        ylabel="Avg. Contralateral Reduction (μ_D)",
        title="Average Contralateral Reduction Across Filters",
        out_path=os.path.join(comparison_dir, "contralateral_reduction_bar.png")
    )

    # Compute average ipsilateral flatness (mean 1/sigma_T_ipsi)
    flatness = {
        "Adjugate Filter": compute_average_ipsilateral_flatness(adj),
        "RR Filter": compute_average_ipsilateral_flatness(rr),
        "VRRR Filter": compute_average_ipsilateral_flatness(vrrr),
    }
    plot_bar_comparison(
        flatness,
        ylabel="Avg. Ipsilateral Flatness (1/σ_T_ipsi)",
        title="Average Ipsilateral Flatness Across Filters",
        out_path=os.path.join(comparison_dir, "ipsilateral_flatness_bar.png")
    )

    # C vs azimuth line plot for all three
    az = sorted(set(adj.keys()) & set(rr.keys()) & set(vrrr.keys()))
    C_data = {
        "Adjugate Filter": [adj[a]['C'] for a in az],
        "RR Filter": [rr[a]['C'] for a in az],
        "VRRR Filter": [vrrr[a]['C'] for a in az],
    }
    plot_C_vs_azimuth_comparison(
        az,
        C_data,
        out_path=os.path.join(comparison_dir, "C_vs_azimuth_comparison.png")
    )

# If run as script, run comparison
if __name__ == "__main__":
    # Generate Adjugate Filter results
    generate_static_results(xtc.generate_filter, "results/default_filter", "Adjugate Filter")
    generate_rotational_results(xtc.generate_filter, "results/default_filter", "Adjugate Filter")

    # Generate RR Filter results
    generate_static_results(xtc.generate_kn_filter, "results/kn_filter", "RR Filter")
    generate_rotational_results(xtc.generate_kn_filter, "results/kn_filter", "RR Filter")

    # Generate VRRR Filter rotational results only
    generate_kn_filter_smart_rotational_only()

    # Now that all data has been created, run the comparison

    run_filter_comparison()