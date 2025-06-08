import numpy as np
import xtc
from tqdm import tqdm
import os
from math import sin, cos, radians
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
logging.basicConfig(level=logging.DEBUG)

def generate_kn_filter_smart_local(
    h_LL, h_LR, h_RL, h_RR,
    speaker_positions, head_position, head_angle=None,
    filter_length=2048, samplerate=48000, debug=False, lambda_freq=1e-4, format='FrequencyDomain'
):
    # Copied from xtc.generate_kn_filter_smart (no import)
    import numpy as np
    from scipy.fft import fft, fftfreq
    import logging
    from scipy.optimize import minimize
    from scipy.interpolate import interp1d
    c = 343.0
    if debug:
        print(f"[DEBUG] Preparing H matrix for angle...")
    H_mat = np.array([[h_LL, h_LR], [h_RL, h_RR]])
    N = 2 ** int(np.ceil(np.log2(len(H_mat[0][0]) + filter_length - 1)))
    omega = 2 * np.pi * fftfreq(N, d=1 / samplerate)
    # Step 1: FFT of HRIRs
    if debug:
        print(f"[DEBUG] Performing FFT...")
    H_f = np.zeros((2, 2, N), dtype=complex)
    for i in range(2):
        for j in range(2):
            H_f[i, j, :] = fft(H_mat[i][j], n=N)
    delay_samples_l = int(np.round(np.linalg.norm(speaker_positions[0] - head_position) / c * samplerate))
    delay_samples_r = int(np.round(np.linalg.norm(speaker_positions[1] - head_position) / c * samplerate))
    if delay_samples_l == delay_samples_r:
        tau_L = 0
        tau_R = 0
    elif delay_samples_l > delay_samples_r:
        if debug:
            print("right source arrives first, applying delay to right")
        tau_L = 0
        tau_R = (delay_samples_l - delay_samples_r) / samplerate
        if debug:
            print("applied {} seconds of delay to right".format(tau_R))
    elif delay_samples_r > delay_samples_l:
        if debug:
            print("left source arrives first, applying delay to left")
        tau_L = (delay_samples_r - delay_samples_l) / samplerate
        tau_R = 0
        if debug:
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
        import numpy as np
        # Interpolate log-scale lambda across +ve frequency
        positive_freqs = freqs[freqs > 0]
        control_freqs = np.geomspace(positive_freqs[0], positive_freqs[-1], len(log_lambdas))
        lambda_interp_func = interp1d(np.log10(control_freqs), log_lambdas, kind='linear', fill_value="extrapolate")
        log_lambda_w = np.full_like(freqs, fill_value=np.nan, dtype=float)
        # Explicitly fill log_lambda_w for positive freqs
        for i, f in enumerate(freqs):
            if f > 0:
                log_lambda_w[i] = lambda_interp_func(np.log10(f))
        lambda_w = 10 ** log_lambda_w
        # For non-positive frequencies, fill with the first positive value or default
        pos_indices = np.where(freqs > 0)[0]
        if len(pos_indices) > 0:
            lambda_w[freqs <= 0] = lambda_w[pos_indices[0]]
        else:
            lambda_w[:] = 1e-4
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
    if debug:
        print(f"[DEBUG] Starting regularization optimization...")
    freqs = np.fft.fftfreq(N, d=1 / samplerate)
    freq_mask = (freqs >= 100) & (freqs <= 7000)
    init_log_lambda = np.full(8, np.log10(1e-4))
    from scipy.optimize import minimize
    res = minimize(
        compute_regularization_error,
        init_log_lambda,
        args=(H_f, freqs, freq_mask),
        method='L-BFGS-B',
        bounds=[(-8, 2)] * 8,
        options={"maxiter": 25}
    )
    if debug:
        print(f"[DEBUG] Optimization complete, building filters...")
    optimized_log_lambda = res.x
    positive_freqs = freqs[freqs > 0]
    control_freqs = np.geomspace(positive_freqs[0], positive_freqs[-1], len(optimized_log_lambda))
    lambda_interp_func = interp1d(np.log10(control_freqs), optimized_log_lambda, kind='linear', fill_value="extrapolate")
    log_lambda_w = np.full_like(freqs, fill_value=np.nan, dtype=float)
    for i, f in enumerate(freqs):
        if f > 0:
            log_lambda_w[i] = lambda_interp_func(np.log10(f))
        else:
            log_lambda_w[i] = 0
    lambda_w = 10 ** log_lambda_w
    # For non-positive frequencies, fill with the first positive value or default
    pos_indices = np.where(freqs > 0)[0]
    if len(pos_indices) > 0:
        lambda_w[freqs <= 0] = lambda_w[pos_indices[0]]
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
    if debug:
        print(f"[DEBUG] Filter generation done.")
    return FLL, FLR, FRL, FRR  # fll, flr, frl, frr


def generate_filter_for_angle(angle, head_position, speaker_positions, sofa_filepath):
    import logging
    print(f"[DEBUG] Starting generation for angle {angle}")
    left_az_deg = -30.0
    right_az_deg = 30.0
    head_angle_rad = radians(angle)
    left_inc_angle = (left_az_deg + angle) % 360
    right_inc_angle = (right_az_deg + angle) % 360
    try:
        HRIR_LL, HRIR_RL, sample_rate_l = extract_hrirs_sam(sofa_filepath, left_inc_angle, show_plots=False, attempt_interpolate=True)
        HRIR_LR, HRIR_RR, sample_rate_r = extract_hrirs_sam(sofa_filepath, right_inc_angle, show_plots=False, attempt_interpolate=True)
        print(f"[DEBUG] HRIRs loaded for angle {angle} (L_inc={left_inc_angle}, R_inc={right_inc_angle})")
        import numpy.ma as ma
        for name, hrir in [('HRIR_LL', HRIR_LL), ('HRIR_LR', HRIR_LR), ('HRIR_RL', HRIR_RL), ('HRIR_RR', HRIR_RR)]:
            if ma.isMaskedArray(hrir):
                num_masked = np.ma.count_masked(hrir)
                print(f"[DEBUG] {name} is a masked array with {num_masked} masked values")
        try:
            FLL, FLR, FRL, FRR = generate_kn_filter_smart_local(
                h_LL=HRIR_LL, h_LR=HRIR_LR, h_RL=HRIR_RL, h_RR=HRIR_RR,
                speaker_positions=speaker_positions,
                head_position=head_position,
                filter_length=2048,
                samplerate=sample_rate_l,
                lambda_freq=1e-4,
                format='FrequencyDomain',
                debug=False
            )
        except Exception as e:
            print(f"[ERROR] Failed to generate filter for key {int(left_inc_angle)}_{int(right_inc_angle)}: {e}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to load HRIRs for angle {angle}: {e}")
        return None
    cache_key = f"{int(angle)}"
    return (cache_key, {'FLL': FLL, 'FLR': FLR, 'FRL': FRL, 'FRR': FRR})


# --- Top-level task function for multiprocessing ---
def task(angle, hrir_path, filter_length, samplerate, lambda_freq):
    """
    Compute the head rotation filter for a given angle.
    """
    import numpy as np
    from xtc import extract_hrirs_sam
    import logging
    logger = logging.getLogger(__name__)
    try:
        # Define the azimuths for the speakers relative to head rotation
        left_az_deg = -30.0
        right_az_deg = 30.0
        left_inc_angle = (left_az_deg + angle) % 360
        right_inc_angle = (right_az_deg + angle) % 360
        # Extract HRIRs for these angles
        hrir_ll, hrir_lr, hrir_rl, hrir_rr = extract_hrirs_sam(
            hrir_path,
            left_inc_angle,
            right_inc_angle,
            attempt_interpolate=True
        )
        # Set up speaker and head positions (example: at origin, 1.5m radius)
        c = 343.0
        distance_l = 1.5
        distance_r = 1.5
        head_position = np.array([0.0, 0.0])
        xL = head_position[0] + distance_l * np.sin(np.radians(-30))
        yL = head_position[1] + distance_l * np.cos(np.radians(-30))
        xR = head_position[0] + distance_r * np.sin(np.radians(30))
        yR = head_position[1] + distance_r * np.cos(np.radians(30))
        speaker_positions = np.array([[xL, yL], [xR, yR]])
        # Generate the filter
        FLL, FLR, FRL, FRR = generate_kn_filter_smart_local(
            hrir_ll, hrir_lr, hrir_rl, hrir_rr,
            speaker_positions, head_position,
            filter_length=filter_length,
            samplerate=samplerate,
            lambda_freq=lambda_freq,
            format='FrequencyDomain'
        )
        return str(int(angle)), {
            'FLL': FLL,
            'FLR': FLR,
            'FRL': FRL,
            'FRR': FRR
        }
    except Exception as e:
        logger.error(f"Failed to generate filter for angle {angle}: {e}")
        return str(int(angle)), None


def process_angle(angle, filepath, filter_length, samplerate):
    try:
        head_angle = angle
        left_az = (head_angle - 30) % 360
        right_az = (head_angle + 30) % 360
        logging.debug(f"Extracting HRIRs for angle {angle} with left azimuth {left_az} and right azimuth {right_az}")
        from xtc import extract_hrirs_sam
        h_LL, h_LR, sr1 = extract_hrirs_sam(filepath, left_az)
        h_RL, h_RR, sr2 = extract_hrirs_sam(filepath, right_az)
        assert sr1 == sr2
        
                # Define the azimuths for the speakers relative to head rotation        # Set up speaker and head positions (example: at origin, 1.5m radius)
        c = 343.0
        distance_l = 1.5
        distance_r = 1.5
        head_position = np.array([0.0, 0.0])
        xL = head_position[0] + distance_l * np.sin(np.radians(-30))
        yL = head_position[1] + distance_l * np.cos(np.radians(-30))
        xR = head_position[0] + distance_r * np.sin(np.radians(30))
        yR = head_position[1] + distance_r * np.cos(np.radians(30))
        speaker_positions = np.array([[xL, yL], [xR, yR]])


        filters = generate_kn_filter_smart_local(h_LL, h_LR, h_RL, h_RR,
                                                 speaker_positions=speaker_positions,
                                                 head_position=head_position,
                                                 filter_length=filter_length,
                                                 samplerate=samplerate,
                                                 debug=True,
                                                 lambda_freq=1e-4)
        return angle, filters
    except Exception as e:
        logging.warning(f"Failed to generate filters for angle {angle}: {e}")
        return angle, None


def precompute_filters():
    filepath = "/Users/willblayney/Development/FYP/XTCLab/P0275_FreeFieldComp_48kHz.sofa"
    angles = list(range(-180, 181))
    filter_length = 2048
    samplerate = 48000
    results = {}

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_angle = {
            executor.submit(process_angle, angle, filepath, filter_length, samplerate): angle
            for angle in angles
        }

        for future in tqdm(as_completed(future_to_angle), total=len(angles), desc="Precomputing filters"):
            angle = future_to_angle[future]
            result = future.result()
            if result and result[1] is not None:
                results[str(result[0])] = result[1]

    np.savez("head_rotation_filter_cache.npz", **results)
    logging.info("[INFO] Precomputed filters saved to head_rotation_filter_cache.npz")


if __name__ == '__main__':
    precompute_filters()