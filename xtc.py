import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import pysofaconventions as sofa
import os
from scipy.fft import fft, ifft, fftfreq
from netCDF4 import Dataset
import time

def compute_speaker_positions_2D(distance=1.5, left_az_deg=-30, right_az_deg=30):
    """
    Return 2x2 array of [ [x_left, y_left], [x_right, y_right] ],
    placing each speaker at the given 'distance' from origin,
    at the specified azimuth angles (in degrees), measured from
    the +y axis, rotating clockwise for negative angles.
    
    '0 deg' is front (y>0), so we do:
      x = r * sin(az_rad)
      y = r * cos(az_rad)
    """
    left_rad = np.radians(left_az_deg)
    right_rad = np.radians(right_az_deg)

    # Left speaker
    xL = distance * np.sin(left_rad)
    yL = distance * np.cos(left_rad)

    # Right speaker
    xR = distance * np.sin(right_rad)
    yR = distance * np.cos(right_rad)

    return np.array([[xL, yL], [xR, yR]])

def plot_hrtf_smaart_style(data, samplerate, azimuth=0):
    """
    Plot the IR pair (left and right ears), their TF magnitudes, and the Smaart-style interaural phase difference (IPD).

    Parameters:
        data (dict): The HRIR data dictionary containing "H_LL", "H_LR", "H_RL", "H_RR", and "samplerate".
        samplerate (int): The sampling rate of the HRIR.
        azimuth (float): The azimuth angle for which to plot the IRs (not used directly, for reference only).
    """
    # Extract left and right IRs
    ir_left = data["H_LL"]
    ir_right = data["H_RR"]

    # Compute delays for each IR by finding the peak of the IR
    delay_left = np.argmax(np.abs(ir_left))
    delay_right = np.argmax(np.abs(ir_right))

    # Align the IRs by removing the delays (circular shift)
    ir_left_aligned = np.roll(ir_left, -delay_left)
    ir_right_aligned = np.roll(ir_right, -delay_right)

    # Compute FFTs (Transfer Functions)
    tf_left = fft(ir_left_aligned)
    tf_right = fft(ir_right_aligned)
    freqs = np.fft.fftfreq(len(ir_left), 1 / samplerate)

    # Keep only positive frequencies
    positive_freqs = freqs[:len(freqs) // 2]
    tf_left = tf_left[:len(freqs) // 2]
    tf_right = tf_right[:len(freqs) // 2]

    # Compute magnitude (no normalization)
    magnitude_left = np.abs(tf_left)
    magnitude_right = np.abs(tf_right)

    # Compute phase difference and correct for delays
    phase_left = np.angle(tf_left)
    phase_right = np.angle(tf_right)
    phase_difference = phase_left - phase_right

    # Wrap phase difference to [-180, 180] degrees
    phase_difference_degrees = np.degrees(phase_difference)
    phase_difference_wrapped = ((phase_difference_degrees + 180) % 360) - 180  # Wrap to [-180, 180]

    # Plot the IRs, TF magnitude, and Smaart-style phase difference
    plt.figure(figsize=(12, 9))

    # Subplot 1: Impulse Responses
    plt.subplot(3, 1, 1)
    time = np.arange(len(ir_left)) / samplerate
    plt.plot(time, ir_left, label="Left Ear IR")
    plt.plot(time, ir_right, label="Right Ear IR", linestyle='--')
    plt.title(f"Impulse Responses at {azimuth}° Azimuth")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Subplot 2: Magnitude Response
    plt.subplot(3, 1, 2)
    plt.plot(positive_freqs, 20 * np.log10(magnitude_left), label="Left Ear Magnitude (dB)")
    plt.plot(positive_freqs, 20 * np.log10(magnitude_right), label="Right Ear Magnitude (dB)", linestyle='--')
    plt.title(f"Magnitude Response at {azimuth}° Azimuth")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()

    # Subplot 3: Smaart-Style Phase Difference (IPD)
    plt.subplot(3, 1, 3)
    plt.plot(positive_freqs, phase_difference_wrapped, label="Phase Difference (Wrapped)")
    plt.title(f"Interaural Phase Difference (IPD) at {azimuth}° Azimuth")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase Difference (degrees)")
    plt.ylim(-180, 180)  # Ensure the y-axis is constrained to -180 to 180 degrees
    plt.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

def plot_hrtf_smaart_style_2x2(data, samplerate, azimuth=0):
    """
    Plot the 2x2 HRTF measurements: impulse responses, magnitude responses, and phase differences.

    Parameters:
        data (dict): The HRIR data dictionary containing "H_LL", "H_LR", "H_RL", "H_RR", and "samplerate".
        samplerate (int): The sampling rate of the HRIR.
        azimuth (float): The azimuth angle for which to plot the IRs (not used directly, for reference only).
    """
    # Extract impulse responses
    H_LL = data["H_LL"]
    H_LR = data["H_LR"]
    H_RL = data["H_RL"]
    H_RR = data["H_RR"]

    # Compute delays for alignment
    delay_LL = np.argmax(np.abs(H_LL))
    delay_LR = np.argmax(np.abs(H_LR))
    delay_RL = np.argmax(np.abs(H_RL))
    delay_RR = np.argmax(np.abs(H_RR))

    # Align impulse responses by removing delays
    H_LL_aligned = np.roll(H_LL, -delay_LL)
    H_LR_aligned = np.roll(H_LR, -delay_LR)
    H_RL_aligned = np.roll(H_RL, -delay_RL)
    H_RR_aligned = np.roll(H_RR, -delay_RR)

    # Compute FFTs (Transfer Functions)
    TF_LL = fft(H_LL_aligned)
    TF_LR = fft(H_LR_aligned)
    TF_RL = fft(H_RL_aligned)
    TF_RR = fft(H_RR_aligned)
    freqs = fftfreq(len(H_LL), 1 / samplerate)

    # Keep only positive frequencies
    positive_freqs = freqs[:len(freqs) // 2]
    TF_LL = TF_LL[:len(freqs) // 2]
    TF_LR = TF_LR[:len(freqs) // 2]
    TF_RL = TF_RL[:len(freqs) // 2]
    TF_RR = TF_RR[:len(freqs) // 2]

    # Compute magnitudes (dB scale)
    mag_LL = 20 * np.log10(np.abs(TF_LL) + 1e-9)
    mag_LR = 20 * np.log10(np.abs(TF_LR) + 1e-9)
    mag_RL = 20 * np.log10(np.abs(TF_RL) + 1e-9)
    mag_RR = 20 * np.log10(np.abs(TF_RR) + 1e-9)

    # Compute phase differences
    phase_LL = np.angle(TF_LL)
    phase_LR = np.angle(TF_LR)
    phase_RL = np.angle(TF_RL)
    phase_RR = np.angle(TF_RR)

    # Plotting
    plt.figure(figsize=(14, 12))

    # Subplot 1: Impulse Responses
    plt.subplot(3, 1, 1)
    time = np.arange(len(H_LL)) / samplerate
    plt.plot(time, H_LL_aligned, label="Speaker L → Ear L")
    plt.plot(time, H_LR_aligned, label="Speaker L → Ear R", linestyle="--")
    plt.plot(time, H_RL_aligned, label="Speaker R → Ear L", linestyle=":")
    plt.plot(time, H_RR_aligned, label="Speaker R → Ear R", linestyle="-.")
    plt.title(f"Impulse Responses at {azimuth}° Azimuth")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Subplot 2: Magnitude Responses
    plt.subplot(3, 1, 2)
    plt.plot(positive_freqs, mag_LL, label="Speaker L → Ear L")
    plt.plot(positive_freqs, mag_LR, label="Speaker L → Ear R", linestyle="--")
    plt.plot(positive_freqs, mag_RL, label="Speaker R → Ear L", linestyle=":")
    plt.plot(positive_freqs, mag_RR, label="Speaker R → Ear R", linestyle="-.")
    plt.title(f"Magnitude Responses at {azimuth}° Azimuth")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()

    # Subplot 3: Phase Responses
    plt.subplot(3, 1, 3)
    plt.plot(positive_freqs, np.degrees(phase_LL), label="Speaker L → Ear L")
    plt.plot(positive_freqs, np.degrees(phase_LR), label="Speaker L → Ear R", linestyle="--")
    plt.plot(positive_freqs, np.degrees(phase_RL), label="Speaker R → Ear L", linestyle=":")
    plt.plot(positive_freqs, np.degrees(phase_RR), label="Speaker R → Ear R", linestyle="-.")
    plt.title(f"Phase Responses at {azimuth}° Azimuth")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (degrees)")
    plt.legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def find_sofa_index_for_azimuth(sf, target_az_deg, tolerance=2.5):
    source_positions = sf.getVariableValue('SourcePosition')  # (N,3)
    target_az_deg = target_az_deg % 360
    best_idx = None
    best_diff = float('inf')
    for i, pos in enumerate(source_positions):
        azimuth = pos[0] % 360
        diff = min(abs(azimuth - target_az_deg), 360 - abs(azimuth - target_az_deg))
        if diff < best_diff and diff <= tolerance:
            best_idx = i
            best_diff = diff
    return best_idx

def extract_4_ir_sofa(filepath, left_az=-30.0, right_az=30.0):
    """
    Extract the four IRs (H_LL, H_LR, H_RL, H_RR) from the angles left_az, right_az.
    """
    sf = sofa.SOFAFile(filepath, 'r')
    hrir = sf.getDataIR()  # shape: (N, M, L)
    samplerate = sf.getSamplingRate()
    source_positions = sf.getVariableValue('SourcePosition')  # (N, 3)
    azimuths = source_positions[:, 0]

    def interpolate_ir(target_az):
        # Normalize azimuths to 0–360
        wrapped_az = np.mod(azimuths, 360)
        target_az = target_az % 360

        diffs = wrapped_az - target_az
        diffs[diffs > 180] -= 360
        diffs[diffs < -180] += 360
        sorted_indices = np.argsort(np.abs(diffs))

        idx1, idx2 = sorted_indices[:2]
        az1, az2 = wrapped_az[idx1], wrapped_az[idx2]
        w2 = abs(target_az - az1) / (abs(az2 - az1) + 1e-9)
        w1 = 1.0 - w2

        hrir_1 = np.asarray(np.ma.filled(hrir[idx1], 0.0), dtype=np.float64)
        hrir_2 = np.asarray(np.ma.filled(hrir[idx2], 0.0), dtype=np.float64)

        # Align by delay (peak position) for both ears
        aligned = []
        for ear in range(2):
            h1 = hrir_1[ear]
            h2 = hrir_2[ear]

            peak1 = np.argmax(np.abs(h1))
            peak2 = np.argmax(np.abs(h2))
            delay = peak1 - peak2
            if delay > 0:
                h2 = np.roll(h2, delay)
            elif delay < 0:
                h1 = np.roll(h1, -delay)

            H1 = np.fft.fft(h1)
            H2 = np.fft.fft(h2)
            H_interp = w1 * H1 + w2 * H2
            h_interp = np.fft.ifft(H_interp).real
            aligned.append(h_interp)
        # check whether arrays are masked:
        
        return np.stack(aligned, axis=0).astype(np.float64)

    H_LL = interpolate_ir(left_az)[0]
    H_LR = interpolate_ir(left_az)[1]
    H_RL = interpolate_ir(right_az)[0]
    H_RR = interpolate_ir(right_az)[1]
    if np.ma.is_masked(H_LL) or np.ma.is_masked(H_LR) or np.ma.is_masked(H_RL) or np.ma.is_masked(H_RR):
        print("masked found")

    sf.close()
    return {
        "H_LL": H_LL,
        "H_LR": H_LR,
        "H_RL": H_RL,
        "H_RR": H_RR,
        "samplerate": samplerate
    }

def save_to_sofa(filename, H_LL, H_LR, H_RL, H_RR, samplerate):
    """
    Save HRIRs to a SOFA file in the SimpleFreeFieldHRIR convention.

    Parameters:
    - filename: str, name of the output SOFA file
    - H_LL, H_LR, H_RL, H_RR: numpy arrays, HRIRs for the four paths
    - samplerate: int, sampling rate of the HRIRs
    """
    import os
    if os.path.exists(filename):
        os.remove(filename)

    sofa_obj = Dataset(filename, mode="w", format="NETCDF4")

    # Annoying Required Attributes
    sofa_obj.Conventions = "SOFA"
    sofa_obj.Version = "1.0"
    sofa_obj.SOFAConventions = "SimpleFreeFieldHRIR"
    sofa_obj.SOFAConventionsVersion = "0.3"
    sofa_obj.APIName = "pysofaconventions"
    sofa_obj.APIVersion = "0.3"
    sofa_obj.AuthorContact = ""
    sofa_obj.Organization = ""
    sofa_obj.License = ""
    sofa_obj.Title = "Simulated HRIR"
    sofa_obj.DataType = "FIR"
    sofa_obj.RoomType = "free field"
    sofa_obj.DateCreated = time.ctime()
    sofa_obj.DateModified = time.ctime()

    # Required Dimensions
    num_measurements = 2  # Left and Right speaker
    num_receivers = 2  # Left and Right ear
    num_samples = len(H_LL)  # HRIR length
    num_coordinates = 3  # Cartesian coordinates (x, y, z)

    # Define dimensions
    sofa_obj.createDimension("M", num_measurements)  # Measurements
    sofa_obj.createDimension("R", num_receivers)     # Receivers
    sofa_obj.createDimension("N", num_samples)       # Samples
    sofa_obj.createDimension("C", num_coordinates)   # Cartesian coordinates
    sofa_obj.createDimension("I", 1)                 # Listener or Source

    # Variables
    ListenerPosition = sofa_obj.createVariable("ListenerPosition", "f8", ("I", "C"))
    ListenerPosition.Units = "metre"
    ListenerPosition.Type = "cartesian"
    ListenerPosition[:] = np.array([[0.0, 0.0, 0.0]])

    ListenerUp = sofa_obj.createVariable("ListenerUp", "f8", ("I", "C"))
    ListenerUp.Units = "metre"
    ListenerUp.Type = "cartesian"
    ListenerUp[:] = np.array([[0.0, 0.0, 1.0]])

    ListenerView = sofa_obj.createVariable("ListenerView", "f8", ("I", "C"))
    ListenerView.Units = "metre"
    ListenerView.Type = "cartesian"
    ListenerView[:] = np.array([[0.0, 1.0, 0.0]])

    # Save the source positions in spherical coordinates
    # Spherical: [azimuth (degrees), elevation (degrees), distance (meters)]
    SourcePosition = sofa_obj.createVariable("SourcePosition", "f8", ("M", "C"))
    SourcePosition.Units = "degree, degree, metre"
    SourcePosition.Type = "spherical"
    SourcePosition[:] = np.array([
        [-30.0, 0.0, 1.5],  # Left speaker at -30 degrees azimuth
        [30.0, 0.0, 1.5],   # Right speaker at 30 degrees azimuth
    ])

    Data_IR = sofa_obj.createVariable("Data.IR", "f8", ("M", "R", "N"))
    Data_IR.ChannelOrdering = "acn"
    Data_IR.Normalization = "sn3d"
    Data_IR[:] = np.array([
        [H_LL, H_LR],  # Left speaker (to both ears)
        [H_RL, H_RR],  # Right speaker (to both ears)
    ])

    Data_SamplingRate = sofa_obj.createVariable("Data.SamplingRate", "f8", ("I"))
    Data_SamplingRate.Units = "hertz"
    Data_SamplingRate[:] = [samplerate]

    sofa_obj.close()
    print(f"SOFA file saved as {filename}")

def inspect_sofa_file(filepath):
    """
    Inspect a SOFA file for source positions, receiver positions, and IR data.
    """
    sf = sofa.SOFAFile(filepath, 'r')
    source_positions = sf.getVariableValue('SourcePosition')  # (N,3)
    receiver_positions = sf.getVariableValue('ReceiverPosition')  # (M,3)
    data_ir = sf.getDataIR()  # shape: (N, M, L)
    samplerate = sf.getSamplingRate()

    print(f"\nFile: {filepath}")
    print(f"Samplerate: {samplerate} Hz")
    print(f"SourcePositions shape: {source_positions.shape}")
    print(f"ReceiverPositions shape: {receiver_positions.shape}")
    print(f"IR data shape: {data_ir.shape}")

    print("\nSource Positions (Azimuth, Elevation, Distance):")
    for i, pos in enumerate(source_positions):
        print(f"  Index {i}: Az={pos[0]:.1f}°, El={pos[1]:.1f}°, Dist={pos[2]:.2f} m")

    sf.close()


def find_sofa_index_for_azimuth(sf, target_az_deg, tolerance=2.5):
    """
    Find the index in the SOFA file closest to a given azimuth (within tolerance).
    This function normalizes azimuth angles to the range [0, 360) to ensure
    equivalence between angles such as -30° and 330°.

    Parameters:
        sf: The SOFAFile object.
        target_az_deg (float): Target azimuth angle in degrees.
        tolerance (float): Tolerance in degrees for matching.

    Returns:
        int: Index of the closest azimuth, or None if no match is found.
    """
    source_positions = sf.getVariableValue('SourcePosition')  # (N,3)
    
    # Normalize the target azimuth to [0, 360)
    target_az_deg = target_az_deg % 360

    best_idx = None
    best_diff = float('inf')
    
    for i, pos in enumerate(source_positions):
        azimuth = pos[0] % 360  # Normalize the azimuth in the SOFA file
        diff = min(abs(azimuth - target_az_deg), 360 - abs(azimuth - target_az_deg))  # Account for circular wrapping

        if diff < best_diff and diff <= tolerance:
            best_idx = i
            best_diff = diff

    return best_idx

def extract_4_ir_sofa(filepath, left_az=-30.0, right_az=30.0):
    """
    Extract the four IRs (H_LL, H_LR, H_RL, H_RR) from the angles left_az, right_az.
    """
    sf = sofa.SOFAFile(filepath, 'r')
    hrir = sf.getDataIR()  # shape: (N, M, L)
    samplerate = sf.getSamplingRate()

    left_idx = find_sofa_index_for_azimuth(sf, left_az)
    right_idx = find_sofa_index_for_azimuth(sf, right_az)
    if left_idx is None or right_idx is None:
        raise ValueError(f"Could not find angles {left_az} or {right_az} in the SOFA file.")

    # L->EarLeft, L->EarRight
    H_LL = hrir[left_idx, 0, :]
    H_LR = hrir[left_idx, 1, :]

    # R->EarLeft, R->EarRight
    H_RL = hrir[right_idx, 0, :]
    H_RR = hrir[right_idx, 1, :]

    sf.close()
    H_LL = np.asarray(H_LL).astype(float)
    H_LR = np.asarray(H_LR).astype(float)
    H_RL = np.asarray(H_RL).astype(float)
    H_RR = np.asarray(H_RR).astype(float)
    return {
        "H_LL": H_LL,  # shape (L,)
        "H_LR": H_LR,
        "H_RL": H_RL,
        "H_RR": H_RR,
        "samplerate": samplerate
    }


def generate_xtc_filters_from_geometry(
    speaker_positions,
    head_position,
    ear_offset=0.15,
    samplerate=48000,
    num_taps=1024,
    reflection_decay=0.9
):
    """
    Generate ideal IRs (H_LL, H_LR, H_RL, H_RR) based on geometry, using delta functions with delay and amplitude.
 
    Parameters:
        speaker_positions: np.ndarray shape (2, 2), positions of Left and Right speakers (x, y)
        head_position: np.ndarray shape (2,), position of head center (x, y)
        ear_offset: float, distance between ears (meters)
        samplerate: int, sampling rate
        num_taps: int, number of samples in each IR
        reflection_decay: float, optional decay factor for repeated delta trains (set to 0 to disable)
 
    Returns:
        dict with H_LL, H_LR, H_RL, H_RR, samplerate
    """
    import numpy as np
 
    c = 343.0  # speed of sound in m/s
    L_ear = head_position + np.array([-ear_offset / 2, 0.0])
    R_ear = head_position + np.array([ ear_offset / 2, 0.0])
 
    def compute_ir(speaker_pos, ear_pos):
        distance = np.linalg.norm(speaker_pos - ear_pos)
        delay_samples = int(round((distance / c) * samplerate))
        amplitude = 1.0 / (distance ** 2 + 1e-9)
        ir = np.zeros(num_taps)
        for n in range(0, num_taps, 2 * delay_samples if delay_samples > 0 else num_taps):
            idx = n
            if idx < num_taps:
                ir[idx] += amplitude * (reflection_decay ** (n // delay_samples))
        return ir
 
    H_LL = compute_ir(speaker_positions[0], L_ear)
    H_LR = compute_ir(speaker_positions[0], R_ear)
    H_RL = compute_ir(speaker_positions[1], L_ear)
    H_RR = compute_ir(speaker_positions[1], R_ear)
 
    return {
        "H_LL": H_LL,
        "H_LR": H_LR,
        "H_RL": H_RL,
        "H_RR": H_RR,
        "samplerate": samplerate
    }

def generate_xtc_filters_mimo(H_LL, H_LR, H_RL, H_RR, samplerate, regularization):
    L = max(len(H_LL), len(H_LR), len(H_RL), len(H_RR))
    N_fft = 2**int(np.ceil(np.log2(L)))

    h_ll = np.fft.fft(H_LL, n=N_fft)
    h_lr = np.fft.fft(H_LR, n=N_fft)
    h_rl = np.fft.fft(H_RL, n=N_fft)
    h_rr = np.fft.fft(H_RR, n=N_fft)

    freqs = np.abs(np.fft.fftfreq(N_fft, d=1.0 / samplerate))
    eps_array = regularization(freqs)

    F11_freq = np.zeros(N_fft, dtype=np.complex128)
    F12_freq = np.zeros(N_fft, dtype=np.complex128)
    F21_freq = np.zeros(N_fft, dtype=np.complex128)
    F22_freq = np.zeros(N_fft, dtype=np.complex128)

    for k in range(N_fft):
        M = np.array([[h_ll[k], h_lr[k]],
                      [h_rl[k], h_rr[k]]])

        det_M = h_ll[k] * h_rr[k] - h_lr[k] * h_rl[k]
        denom = det_M + eps_array[k]
        if np.abs(denom) > 1e-10:
            Minv = np.array([[h_rr[k], -h_lr[k]],
                             [-h_rl[k], h_ll[k]]]) / denom
        else:
            Minv = np.eye(2) * 1e-3

        F11_freq[k], F12_freq[k] = Minv[0, 0], Minv[0, 1]
        F21_freq[k], F22_freq[k] = Minv[1, 0], Minv[1, 1]

    f11_time = np.fft.ifft(F11_freq).real
    f12_time = np.fft.ifft(F12_freq).real
    f21_time = np.fft.ifft(F21_freq).real
    f22_time = np.fft.ifft(F22_freq).real

    return f11_time, f12_time, f21_time, f22_time

def plot_ir(ir_data, samplerate, title):
    """
    Plot impulse responses.
    """
    time = np.arange(len(ir_data)) / samplerate
    plt.plot(time, ir_data, label=title)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()


def visualize_4_ir(data):
    """
    Visualize the extracted IRs.
    """
    samplerate = data["samplerate"]
    plt.figure(figsize=(10, 8))
    plot_ir(data["H_LL"], samplerate, "Speaker L → Ear L")
    plot_ir(data["H_LR"], samplerate, "Speaker L → Ear R")
    plot_ir(data["H_RL"], samplerate, "Speaker R → Ear L")
    plot_ir(data["H_RR"], samplerate, "Speaker R → Ear R")
    plt.tight_layout()
    plt.show()


def check_symmetric_angles(filepath, left_az=-30.0, right_az=30.0, tolerance=2.5):
    """
    Check if the SOFA file contains measurements for symmetric angles.
    """
    sf = sofa.SOFAFile(filepath, 'r')
    left_idx = find_sofa_index_for_azimuth(sf, left_az, tolerance)
    right_idx = find_sofa_index_for_azimuth(sf, right_az, tolerance)
    sf.close()

    if left_idx is not None and right_idx is not None:
        print(f"Found symmetric pair: {left_az}° and {right_az}°.")
        return True
    else:
        print(f"Missing required angles: {left_az}° or {right_az}°.")
        return False


def calibrate_amplitude(playback_channel, recording_channel, output_device, input_device, samplerate, noise_margin_db=10, max_attempts=10, calibration_scale=0.5):
    """
    Calibrate the playback amplitude by playing tones through the speaker and checking the microphone level.

    Parameters:
        playback_channel (int): The output channel (e.g., left or right speaker).
        recording_channel (int): The input channel (e.g., left or right microphone).
        output_device (dict): The selected output device.
        input_device (dict): The selected input device.
        samplerate (int): Sampling rate of the device.
        noise_margin_db (float): Desired dB level above the noise floor (default: 20 dB).
        max_attempts (int): Maximum number of attempts to find the right level.
        calibration_scale (float): Scale factor for the calibration tone amplitude.

    Returns:
        float: The calibrated amplitude for playback.
    """
    print(f"\nCalibrating amplitude for {['Left', 'Right'][playback_channel]} speaker...")

    amplitude = 0.1  # Initial amplitude (10% of max)
    tone_frequency = 1000  # Test tone frequency in Hz
    duration = 0.5  # Test tone duration in seconds
    samples = int(samplerate * duration)
    time_array = np.arange(samples) / samplerate
    test_tone = calibration_scale * np.sin(2 * np.pi * tone_frequency * time_array).astype(np.float32)  # Scaled tone

    # Measure the noise floor
    print("Measuring noise floor...")
    with sd.InputStream(device=input_device['index'], samplerate=samplerate, channels=1, dtype='float32') as stream:
        noise_recording = stream.read(samples)[0].flatten()
        noise_floor_db = 20 * np.log10(np.max(np.abs(noise_recording)) + 1e-9)
    print(f"Measured noise floor: {noise_floor_db:.2f} dBFS")

    target_db = noise_floor_db + noise_margin_db  # Target level above the noise floor
    print(f"Target level: {target_db:.2f} dBFS")

    for attempt in range(max_attempts):
        # Create the output signal
        output_signal = np.zeros((samples, output_device['max_output_channels']), dtype=np.float32)
        output_signal[:, playback_channel] = amplitude * test_tone

        with sd.Stream(device=(output_device['index'], input_device['index']),
                       samplerate=samplerate,
                       channels=(output_device['max_output_channels'], input_device['max_input_channels']),
                       dtype='float32') as stream:
            # Play and record
            stream.write(output_signal)
            recorded = stream.read(samples)[0][:, recording_channel]

        # Calculate peak level in dBFS
        peak_level = 20 * np.log10(np.max(np.abs(recorded)) + 1e-9)  # Add small value to avoid log(0)

        print(f"Attempt {attempt + 1}: Amplitude={amplitude:.2f}, Peak Level={peak_level:.2f} dBFS")

        # Check if the level is within the target range
        if target_db - 1 <= peak_level <= target_db + 1:
            print(f"Calibrated amplitude for {['Left', 'Right'][playback_channel]} speaker: {amplitude:.2f}")
            return amplitude

        # Increase the amplitude for the next attempt
        amplitude *= 1.5
        if amplitude > 1.0:
            print("Amplitude reached the maximum value (clipping risk).")
            break

    print(f"Calibration failed to reach target level after {max_attempts} attempts. Using amplitude: {amplitude:.2f}")
    return amplitude

def align_impulse_responses(H_LL, H_LR, H_RL, H_RR):
    """
    Align the four impulse responses based on the time-of-arrival (peak position).
    This ensures all IRs are shifted to have the peak at the same time index.

    Parameters:
        H_LL, H_LR, H_RL, H_RR: Impulse responses (numpy arrays).

    Returns:
        Aligned impulse responses: H_LL, H_LR, H_RL, H_RR.
    """
    # Find the peak positions
    peak_LL = np.argmax(np.abs(H_LL))
    peak_LR = np.argmax(np.abs(H_LR))
    peak_RL = np.argmax(np.abs(H_RL))
    peak_RR = np.argmax(np.abs(H_RR))

    # Determine the earliest peak position
    earliest_peak = min(peak_LL, peak_LR, peak_RL, peak_RR)

    # Align all responses to the earliest peak
    H_LL_aligned = np.roll(H_LL, -peak_LL + earliest_peak)
    H_LR_aligned = np.roll(H_LR, -peak_LR + earliest_peak)
    H_RL_aligned = np.roll(H_RL, -peak_RL + earliest_peak)
    H_RR_aligned = np.roll(H_RR, -peak_RR + earliest_peak)

    return H_LL_aligned, H_LR_aligned, H_RL_aligned, H_RR_aligned
def compute_rt60(ir, samplerate, noise_floor_region=(0, 0.05)):
    """
    Compute the RT60 (reverberation time) using the Schroeder backward integration method.
    
    Parameters:
        ir (numpy array): Impulse response.
        samplerate (int): Sampling rate of the IR.
        noise_floor_region (tuple): Time range (in seconds) for estimating the noise floor.
    
    Returns:
        float: Estimated RT60 in seconds.
    """
    # Compute energy decay curve (Schroeder integral)
    energy = np.cumsum(ir[::-1] ** 2)[::-1]  # Reverse cumulative sum of squared IR
    energy_db = 10 * np.log10(energy / np.max(energy + 1e-9))  # Normalize and convert to dB

    # Estimate noise floor from the specified region
    start_idx = int(noise_floor_region[0] * samplerate)
    end_idx = int(noise_floor_region[1] * samplerate)
    noise_floor_db = 10 * np.log10(np.mean(ir[start_idx:end_idx] ** 2) + 1e-9)

    # Find RT60 by locating the -60 dB point
    rt60_idx = np.where(energy_db <= noise_floor_db - 60)[0]
    if len(rt60_idx) == 0:
        return None  # Unable to compute RT60
    rt60_time = rt60_idx[0] / samplerate
    print(f"Estimated RT60: {rt60_time:.3f} seconds")
    return rt60_time

def fft_noise_subtraction(ir, noise_floor, samplerate, smoothing=0.1):
    """
    Perform spectral subtraction to remove noise from the IR using its noise floor.
    
    Parameters:
        ir (numpy array): The impulse response.
        noise_floor (numpy array): The noise floor (same length as IR).
        samplerate (int): Sampling rate of the IR.
        smoothing (float): Smoothing factor for the subtraction (0 to 1).

    Returns:
        numpy array: Noise-reduced IR.
    """
    # Compute FFTs
    ir_fft = np.fft.fft(ir)
    noise_fft = np.fft.fft(noise_floor)

    # Subtract noise spectrum (with a smoothing factor)
    mag_ir = np.abs(ir_fft)
    mag_noise = np.abs(noise_fft)
    phase_ir = np.angle(ir_fft)

    # Smoothed subtraction
    reduced_mag = np.maximum(mag_ir - smoothing * mag_noise, 0)
    ir_cleaned_fft = reduced_mag * np.exp(1j * phase_ir)

    # Inverse FFT
    ir_cleaned = np.fft.ifft(ir_cleaned_fft).real
    return ir_cleaned

def post_process_ir_rt60(ir, samplerate, noise_floor_region=(0, 0.05), smoothing_factor=0.1):
    """
    Post-process IR using RT60 windowing and spectral noise subtraction.
    
    Parameters:
        ir (numpy array): The raw impulse response.
        samplerate (int): Sampling rate of the IR.
        noise_floor_region (tuple): Time range (in seconds) for estimating the noise floor.
        smoothing_factor (float): Smoothing factor for spectral subtraction.
    
    Returns:
        numpy array: Processed impulse response.
    """
    # Compute RT60 and apply windowing
    rt60 = compute_rt60(ir, samplerate, noise_floor_region=noise_floor_region)
    if rt60 is None:
        print("RT60 could not be computed. Returning raw IR.")
        return ir

    # Apply windowing based on RT60
    rt60_idx = int(rt60 * samplerate)
    window = np.zeros(len(ir))
    window[:rt60_idx] = tukey(rt60_idx, alpha=0.1)[:rt60_idx]
    ir_windowed = ir * window

    # Estimate noise floor
    start_idx = int(noise_floor_region[0] * samplerate)
    end_idx = int(noise_floor_region[1] * samplerate)
    noise_floor = ir[start_idx:end_idx]
    noise_floor_padded = np.zeros_like(ir)
    noise_floor_padded[:len(noise_floor)] = noise_floor

    # Apply FFT-based noise subtraction
    ir_cleaned = fft_noise_subtraction(ir_windowed, noise_floor_padded, samplerate, smoothing=smoothing_factor)
    return ir_cleaned

def visualize_ir_pre_post(ir_raw, ir_processed, samplerate, title="Impulse Response"):
    """
    Visualize raw and processed impulse responses.
    """
    time_raw = np.arange(len(ir_raw)) / samplerate
    time_processed = np.arange(len(ir_processed)) / samplerate

    plt.figure(figsize=(12, 6))
    plt.plot(time_raw, ir_raw, label="Raw IR", alpha=0.7)
    plt.plot(time_processed, ir_processed, label="Processed IR", alpha=0.9)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def recordHRTF():
    """
    Record HRTFs using a specified input and output device. Includes inline noise sampling,
    spectral subtraction, and RT60-based windowing for denoising during the recording process.
    """
    print("Recording HRTF...")

    # Get the list of available devices
    devices = sd.query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    output_devices = [device for device in devices if device['max_output_channels'] > 0]

    # Print the list of available input devices
    if not input_devices:
        print("No input devices found.")
        return
    print("\nAvailable Input Devices:")
    for idx, device in enumerate(input_devices):
        print(f"{idx + 1}: {device['name']} ({device['max_input_channels']} channels)")

    # Print the list of available output devices
    if not output_devices:
        print("No output devices found.")
        return
    print("\nAvailable Output Devices:")
    for idx, device in enumerate(output_devices):
        print(f"{idx + 1}: {device['name']} ({device['max_output_channels']} channels)")

    # Select the input and output devices
    input_device_idx = int(input("\nSelect the input device index: ")) - 1
    output_device_idx = int(input("Select the output device index: ")) - 1

    # Get the selected input and output devices
    input_device = input_devices[input_device_idx]
    output_device = output_devices[output_device_idx]

    # Print the selected input and output devices
    print(f"\nSelected Input Device: {input_device['name']}")
    print(f"Selected Output Device: {output_device['name']}")

    # Channel selection for inputs
    print("\nInput Channels:")
    for i in range(input_device['max_input_channels']):
        print(f"  {i + 1}: Channel {i + 1}")

    left_mic_channel = int(input("\nSelect the channel for the left microphone: ")) - 1
    right_mic_channel = int(input("Select the channel for the right microphone: ")) - 1

    # Channel selection for outputs
    print("\nOutput Channels:")
    for i in range(output_device['max_output_channels']):
        print(f"  {i + 1}: Channel {i + 1}")

    left_speaker_channel = int(input("\nSelect the channel for the left speaker: ")) - 1
    right_speaker_channel = int(input("Select the channel for the right speaker: ")) - 1

    # Define recording parameters
    global samplerate
    samplerate = int(output_device['default_samplerate'])  # Use the device's default sample rate
    noise_duration = 2.0  # Duration for noise sampling (seconds)
    impulse_duration = 1.0  # Duration of each recording (seconds)
    ir_length = int(samplerate * impulse_duration)  # Length of the impulse response

    # Noise sampling stage
    print("\nSampling noise floor...")
    with sd.InputStream(device=input_device['index'], samplerate=samplerate, channels=2, dtype='float32') as stream:
        noise_samples = stream.read(int(noise_duration * samplerate))[0]
    noise_profile = np.mean(noise_samples, axis=0)  # Averaged noise profile
    print("Noise sampling completed.")

    # Calibrate amplitude for each speaker-microphone pair
    left_amplitude = calibrate_amplitude(left_speaker_channel, left_mic_channel, output_device, input_device, samplerate)
    right_amplitude = calibrate_amplitude(right_speaker_channel, right_mic_channel, output_device, input_device, samplerate)

    impulse_sf = 0.1

    # Create impulse signals
    impulse_left = np.zeros(ir_length, dtype=np.float32)
    impulse_right = np.zeros(ir_length, dtype=np.float32)
    impulse_left[0] = left_amplitude / impulse_sf
    impulse_right[0] = right_amplitude / impulse_sf

    def denoise_recorded_signal(signal, noise_profile, samplerate):
        """
        Perform spectral subtraction to denoise the recorded signal.
        """
        signal_fft = fft(signal)
        noise_fft = fft(noise_profile, n=len(signal))
        signal_mag = np.abs(signal_fft)
        noise_mag = np.abs(noise_fft)
        signal_phase = np.angle(signal_fft)

        # Subtract the noise magnitude with smoothing
        reduced_mag = np.maximum(signal_mag - 0.5 * noise_mag, 0)
        cleaned_signal_fft = reduced_mag * np.exp(1j * signal_phase)
        cleaned_signal = np.fft.ifft(cleaned_signal_fft).real

        # Truncate based on RT60
        rt60 = compute_rt60(cleaned_signal, samplerate)
        if rt60:
            rt60_idx = int(rt60 * samplerate)
            cleaned_signal[rt60_idx:] = 0  # Truncate after RT60
        return cleaned_signal

    def log_and_record(playback_channel, recording_channel, impulse_signal, output_device, input_device, samplerate):
        """
        Play an impulse on a given playback channel and record from a given recording channel.
        """
        print(f"\nPlaying impulse on {['Left', 'Right'][playback_channel]} speaker "
            f"and recording on {['Left', 'Right'][recording_channel]} microphone...")

        # Create the output signal
        samples = len(impulse_signal)
        output_signal = np.zeros((samples, output_device['max_output_channels']), dtype=np.float32)
        output_signal[:, playback_channel] = impulse_signal

        with sd.Stream(device=(output_device['index'], input_device['index']),
                       samplerate=samplerate,
                       channels=(output_device['max_output_channels'], input_device['max_input_channels']),
                       dtype='float32') as stream:
            # Play and record
            stream.write(output_signal)
            recorded = stream.read(samples)[0][:, recording_channel]

        print(f"Recording on {['Left', 'Right'][recording_channel]} microphone completed.")
        return recorded.flatten()

    # Record and process impulse responses
    H_LL = denoise_recorded_signal(
        log_and_record(left_speaker_channel, left_mic_channel, impulse_left, output_device, input_device, samplerate),
        noise_profile, samplerate
    )
    H_LR = denoise_recorded_signal(
        log_and_record(left_speaker_channel, right_mic_channel, impulse_left, output_device, input_device, samplerate),
        noise_profile, samplerate
    )
    H_RL = denoise_recorded_signal(
        log_and_record(right_speaker_channel, left_mic_channel, impulse_right, output_device, input_device, samplerate),
        noise_profile, samplerate
    )
    H_RR = denoise_recorded_signal(
        log_and_record(right_speaker_channel, right_mic_channel, impulse_right, output_device, input_device, samplerate),
        noise_profile, samplerate
    )

    print("\nImpulse responses recorded successfully.")

    # Save the impulse responses to a SOFA file
    sofa_filename = "Recorded_HRTF_Denoised.sofa"
    save_to_sofa(sofa_filename, H_LL, H_LR, H_RL, H_RR, samplerate)

    # Plot the recorded HRTFs
    data = {
        "H_LL": H_LL,
        "H_LR": H_LR,
        "H_RL": H_RL,
        "H_RR": H_RR,
        "samplerate": samplerate
    }
    visualize_4_ir(data)
    plot_hrtf_smaart_style_2x2(data, samplerate, azimuth=0)

def main():
    print("Crosstalk Cancellation System")
    sofa_files = [f for f in os.listdir('.') if f.endswith('.sofa')]

    if not sofa_files:
        print("\nNo .sofa files found in the current directory.")
        return

    print("\nAvailable SOFA files:")
    for idx, file in enumerate(sofa_files):
        print(f"{idx + 1}: {file}")
    file_idx = int(input("\nSelect a SOFA file by index: ")) - 1
    sofa_file = sofa_files[file_idx]

    while True:
        print("\nOptions:")
        print("1. Inspect SOFA File")
        print("2. Check for Symmetric Angles")
        print("3. Extract and Visualize 4 IRs")
        print("4. Record HRTF")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            inspect_sofa_file(sofa_file)
        elif choice == "2":
            left_az = float(input("Enter left speaker azimuth (e.g., -30): "))
            right_az = float(input("Enter right speaker azimuth (e.g., 30): "))
            tolerance = float(input("Enter azimuth tolerance (degrees): "))
            check_symmetric_angles(sofa_file, left_az, right_az, tolerance)
        elif choice == "3":
            data = extract_4_ir_sofa(sofa_file, left_az=-30.0, right_az=30.0)
            visualize_4_ir(data)
            plot_hrtf_smaart_style(data, data["samplerate"], azimuth=0)
        elif choice == "4":
            print("Entering record mode...")
            recordHRTF()
            break
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

def test_xtc_filter_identity():
    """
    Simple test: use identity HRIRs and verify that the XTC filter becomes an identity matrix.
    """
    samplerate = 48000
    L = 512
    delta = np.zeros(L)
    delta[0] = 1.0  # unit impulse

    H_LL = delta.copy()
    H_LR = np.zeros_like(delta)
    H_RL = np.zeros_like(delta)
    H_RR = delta.copy()

    f11, f12, f21, f22 = generate_xtc_filters_mimo(H_LL, H_LR, H_RL, H_RR, samplerate, regularization=1e-6)

    def is_delta(sig):
        return np.allclose(sig[1:], 0, atol=1e-6) and np.isclose(sig[0], 1.0, atol=1e-6)

    assert is_delta(f11), "f11 should be delta"
    assert is_delta(f22), "f22 should be delta"
    assert np.allclose(f12, 0, atol=1e-6), "f12 should be zero"
    assert np.allclose(f21, 0, atol=1e-6), "f21 should be zero"

    print("test_xtc_filter_identity passed.")



if __name__ == "__main__":
    test_xtc_filter_identity()
    main()

