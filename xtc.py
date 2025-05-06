import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import pysofaconventions as sofa
import spatialaudiometrics
import os
from scipy.fft import fft, ifft, fftfreq
from netCDF4 import Dataset
import time

import spatialaudiometrics.load_data
import spatialaudiometrics.visualisation

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

def align_impulse_responses(H_LL, H_LR, H_RL, H_RR, threshold_db=-60.0):
    """
    Remove free-field delays from HRIRs while preserving ITDs.
    Each speaker's HRIRs are shifted based on the direct-path arrival.

    Parameters:
        H_LL, H_LR, H_RL, H_RR: HRIRs as numpy arrays.
        threshold_db: Threshold in dB above noise floor to detect first significant arrival.

    Returns:
        Aligned impulse responses: H_LL_new, H_LR_new, H_RL_new, H_RR_new
    """
    import numpy as np

    def find_first_above_threshold(ir, threshold_db):
        """Find the first sample above the threshold relative to max."""
        energy_db = 20 * np.log10(np.abs(ir) + 1e-12)
        max_db = np.max(energy_db)
        threshold = max_db + threshold_db
        idx = np.argmax(energy_db >= threshold)
        return idx - 10 # shift back 10 samples to avoid clipping pre ringing

    # Group by speaker
    delay_LL = find_first_above_threshold(H_LL, threshold_db)
    delay_RR = find_first_above_threshold(H_RR, threshold_db)

    # For left speaker: align based on LEFT->LEFT
    shift_left = delay_LL
    H_LL_new = np.roll(H_LL, -shift_left)
    H_LR_new = np.roll(H_LR, -shift_left)

    # For right speaker: align based on RIGHT->RIGHT
    shift_right = delay_RR
    H_RL_new = np.roll(H_RL, -shift_right)
    H_RR_new = np.roll(H_RR, -shift_right)

    # Zero out samples that rolled in from the end
    H_LL_new[-shift_left:] = 0.0
    H_LR_new[-shift_left:] = 0.0
    H_RL_new[-shift_right:] = 0.0
    H_RR_new[-shift_right:] = 0.0

    # Debug plot for visual sanity check
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Before Alignment")
    plt.plot(H_LL, label="H_LL")
    plt.plot(H_LR, label="H_LR")
    plt.plot(H_RL, label="H_RL")
    plt.plot(H_RR, label="H_RR")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.title("After Alignment")
    plt.plot(H_LL_new, label="H_LL_new")
    plt.plot(H_LR_new, label="H_LR_new")
    plt.plot(H_RL_new, label="H_RL_new")
    plt.plot(H_RR_new, label="H_RR_new")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return H_LL_new, H_LR_new, H_RL_new, H_RR_new

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
        if diff < best_diff and diff <= tolerance and pos[1] == 0:
            best_idx = i
            best_diff = diff
    return best_idx

def extract_hrirs_sam(filepath, source_az, show_plots = False, attempt_interpolate=False):
    # If we're not attempting to interpolate, round to the nearest 5 degrees
    hrtf = spatialaudiometrics.load_data.HRTF(filepath)
    if source_az < 0:
        source_az += 360

    if not attempt_interpolate:
        source_az = round(source_az / 5) * 5

        idx = np.where((hrtf.locs[:,0] == source_az) & (hrtf.locs[:,1] == 0))[0][0]

        hrir_l = hrtf.hrir[idx,0,:]
        hrir_r = hrtf.hrir[idx,1,:]
    
    else:
        # throw an error - not implemented yet
        raise NotImplementedError("Interpolation not implemented yet.")
    
    if(show_plots):
        fig,gs = spatialaudiometrics.visualisation.create_fig()
        axes = fig.add_subplot(gs[1:6,1:6])
        spatialaudiometrics.visualisation.plot_hrir_both_ears(hrtf,source_az,0,axes)
        axes = fig.add_subplot(gs[1:6,7:12])
        spatialaudiometrics.visualisation.plot_itd_overview(hrtf)
        spatialaudiometrics.visualisation.show()
        plt.show()  # Explicitly call plt.show() to keep the plot open????
    samplerate = sofa.SOFAFile(filepath, 'r').getSamplingRate()
    return hrir_l, hrir_r, samplerate

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
        elevation = pos[1]
        diff = min(abs(azimuth - target_az_deg), 360 - abs(azimuth - target_az_deg))  # Account for circular wrapping

        if diff < best_diff and diff <= tolerance and elevation == 0:
            assert elevation == 0, f"============== ELEVATION NONZERO ==============="
            best_idx = i
            best_diff = diff

    return best_idx

def extract_4_ir_sofa(filepath, left_az=-30.0, right_az=30.0, debug = False):
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
def generate_xtc_filters_mimo_cf(
    H_LL, H_LR, H_RL, H_RR,
    samplerate, head_position, speaker_positions,
    ear_offset=0.15, regularization=None,
    filter_length=4096, delay_compensation=True
):
    """
    Generate XTC filters using regularized frequency-domain inversion (from Mouchtaris et al.).

    Parameters:
        H_LL, H_LR, H_RL, H_RR: HRIRs (assumed free-field compensated)
        samplerate: sample rate (Hz)
        head_position: (x, y) of head center
        speaker_positions: [[xL, yL], [xR, yR]]
        ear_offset: interaural distance (default 0.15m)
        regularization: scalar regularization constant or None (default 0.05)
        filter_length: length of each filter (samples)
        delay_compensation: whether to pad filters based on speaker/ear delays

    Returns:
        f11, f12, f21, f22 (numpy arrays)
    """
    import numpy as np

    c = 343.0  # Speed of sound (m/s)
    b = regularization if regularization is not None else 0.05

    # Step 1: Pad all IRs to same length
    L = max(len(H_LL), len(H_LR), len(H_RL), len(H_RR))
    N_fft = 2 ** int(np.ceil(np.log2(L)))

    def pad(ir):
        return np.pad(ir, (0, N_fft - len(ir)), mode='constant') if len(ir) < N_fft else ir[:N_fft]

    HRIRs = np.zeros((2, 2, N_fft))
    HRIRs[0, 0, :] = pad(H_LL)
    HRIRs[0, 1, :] = pad(H_LR)
    HRIRs[1, 0, :] = pad(H_RL)
    HRIRs[1, 1, :] = pad(H_RR)

    # Step 2: FFT for each HRIR
    C_f = np.zeros((2, 2, N_fft), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            C_f[i, j, :] = np.fft.fft(HRIRs[i, j, :])

    # Step 3: Regularized inversion
    H_f = np.zeros((2, 2, N_fft), dtype=np.complex128)
    for k in range(N_fft):
        C = C_f[:, :, k]
        C_H = C.conj().T
        H_f[:, :, k] = np.linalg.inv(C_H @ C + b * np.eye(2)) @ C_H

    # Step 4: IFFT and fftshift
    H_n = np.zeros((2, 2, N_fft))
    for i in range(2):
        for j in range(2):
            h_time = np.fft.ifft(H_f[i, j, :]).real
            H_n[i, j, :] = np.fft.fftshift(h_time)

    # Step 5: Extract filters and apply window
    def extract_window_center(h, N):
        center = len(h) // 2
        half = N // 2
        extract = h[center - half:center + half]
        return extract * np.hanning(N)

    f11 = extract_window_center(H_n[0, 0, :], filter_length/2)
    f12 = extract_window_center(H_n[0, 1, :], filter_length/2)
    f21 = extract_window_center(H_n[1, 0, :], filter_length/2)
    f22 = extract_window_center(H_n[1, 1, :], filter_length/2)

    # Step 6: Apply relative delays based on geometry
    if delay_compensation:
        L_ear = head_position + np.array([-ear_offset / 2, 0.0])
        R_ear = head_position + np.array([ ear_offset / 2, 0.0])

        def delay_samples(spk, ear):
            return int(round(np.linalg.norm(spk - ear) / c * samplerate))

        delays = {
            'f11': delay_samples(speaker_positions[0], L_ear),
            'f12': delay_samples(speaker_positions[0], R_ear),
            'f21': delay_samples(speaker_positions[1], L_ear),
            'f22': delay_samples(speaker_positions[1], R_ear),
        }

        max_delay = max(delays.values())

        def pad_delay(ir, actual, max_d):
            return np.pad(ir, (max_d - actual, 0), mode='constant')[:filter_length]

        f11 = pad_delay(f11, delays['f11'], max_delay)
        f12 = pad_delay(f12, delays['f12'], max_delay)
        f21 = pad_delay(f21, delays['f21'], max_delay)
        f22 = pad_delay(f22, delays['f22'], max_delay)

    print(f"[DEBUG] Max |f11| = {np.max(np.abs(f11)):.4f}, |f12| = {np.max(np.abs(f12)):.4f}, |f21| = {np.max(np.abs(f21)):.4f}, |f22| = {np.max(np.abs(f22)):.4f}")
    return f11, f12, f21, f22

def generate_filter(h_LL, h_LR, h_RL, h_RR, speaker_positions, head_position, filter_length, samplerate=48000, debug=False):
    L = max(len(h_LL), len(h_LR), len(h_RL), len(h_RR))
    # how many points? k is a hyperparameter. more points = more frequency domain precision at the expense of time domain precision.
    M = filter_length
    print("generate filter called")
    N = 2**int(np.ceil(np.log2(L + M - 1)))
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
        print("Left and right sources are coherent, no delay applied")
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

    if debug:
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

    if debug:
        # Verify HF ≈ I across frequency
        H_freq = np.array([[H_ll, H_lr], [H_rl, H_rr]])
        F_freq = np.array([[F11, F12], [F21, F22]])
        HF = np.einsum('imk,mjk->ijk', H_freq, F_freq)
        identity = np.eye(2)[:, :, None]

        # Compute per-bin maximum deviation and report top 10 “bad” bins
        dev = np.max(np.abs(HF - identity), axis=(0,1))          # shape (N,)
        top_k = np.argsort(dev)[-10:][::-1]                      # indices of 10 largest
        print("[DEBUG] Top 10 HF deviation bins:")
        for k in top_k:
            print(f"  bin {k}: deviation {dev[k]:.3e}")
        # Report H matrix determinant and condition number at worst bins
        print("[DEBUG] H matrix det and condition number at top deviation bins:")
        # Only check HF on well-conditioned bins within 120–16 kHz
        freqs = np.fft.fftfreq(N, d=1.0 / samplerate)
        band_bins = (np.abs(freqs) >= 120) & (np.abs(freqs) <= 16000)
        det = H_ll * H_rr - H_lr * H_rl
        det_thresh = 1e-3 * np.max(np.abs(det))
        good_bins = band_bins & (np.abs(det) > det_thresh)
        for k in top_k:
            Hk = np.array([[H_ll[k], H_lr[k]], [H_rl[k], H_rr[k]]])
            svals = np.linalg.svd(Hk, compute_uv=False)
            cond = svals[0] / svals[-1] if svals[-1] > 0 else np.inf
            print(f"  bin {k}: det={det[k]:.3e}, cond={cond:.3e}, |H_ll|={abs(H_ll[k]):.3f}, |H_lr|={abs(H_lr[k]):.3f}")

        max_dev = dev.max()
        print(f"[DEBUG] generate_filter HF max deviation: {max_dev:.3e}")
        # Compare only well-conditioned bins to the identity matrix (broadcast across bins)
        assert np.allclose(HF[:,:,good_bins], identity, atol=1e-6), \
            f"Frequency-domain HF check failed on well-conditioned bins (max deviation {max_dev:.3e})"

    # Delay compensation
    freqs = np.fft.fftfreq(N, d=1/samplerate)
    omega = 2 * np.pi * freqs
    # delay_samples_l = int(np.round((np.linalg.norm(speaker_positions[0] - head_position) / 343.0) * samplerate))
    # delay_samples_r = int(np.round((np.linalg.norm(speaker_positions[1] - head_position) / 343.0) * samplerate))
    # I was previously rolling by the delay, but this is not correct.
    # Now I'm rolling by a static delay of 200 samples to prevent non-causality.
    delay_samples_l = -400
    delay_samples_r = -400

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
        # Frequency axis (positive half)
        freqs = np.fft.fftfreq(N, d=1.0/samplerate)
        positive = freqs[:N//2]

        # Spectra of delayed H matrix
        specs_H_comp = [HLL_comp, HLR_comp, HRL_comp, HRR_comp]
        # Spectra of embedded-delay filters
        specs_F = [F11, F12, F21, F22]

        # Compute HF = H_comp @ F per bin
        H_freq_comp = np.array([[HLL_comp, HLR_comp],
                                [HRL_comp, HRR_comp]])
        F_freq = np.array([[F11, F12],
                           [F21, F22]])
        HF = np.einsum('imk,mjk->ijk', H_freq_comp, F_freq)
        specs_HF = [HF[0,0,:], HF[0,1,:], HF[1,0,:], HF[1,1,:]]

        # Time-domain IRs
        f_time = [f11, f12, f21, f22]
        t = np.arange(len(f11)) / samplerate

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

    return fll, flr, frl, frr


def generate_xtc_filters_mimo(H_LL, H_LR, H_RL, H_RR, samplerate, head_position, speaker_positions, ear_offset=0.15, regularization=None, filter_length=4096, delay_compensation=True):
    c = 343.0  # speed of sound (m/s)

    # Step 1: Align HRIRs to center
    # H_LL, H_LR, H_RL, H_RR = align_impulse_responses(H_LL=H_LL, H_LR=H_LR, H_RL=H_RL, H_RR=H_RR)

    # A reminder!
    # HRIR_LL = left ear left speaker
    # HRIR_LR = left ear right speaker
    # HRIR_RL = right ear left speaker
    # HRIR_RR = right ear right speaker

    L = max(len(H_LL), len(H_LR), len(H_RL), len(H_RR))
    N_fft = 2**int(np.ceil(np.log2(L)))

    print("generate_xtc_filters_mimo: max length of H recordings:", L)
    print("requested filter length: ", filter_length)
    
    # Step 2: FFT of IRs
    h_ll = np.fft.fft(H_LL, n=N_fft)
    h_lr = np.fft.fft(H_LR, n=N_fft)
    h_rl = np.fft.fft(H_RL, n=N_fft)
    h_rr = np.fft.fft(H_RR, n=N_fft)

    # Step 3: Setup Regularization
    # freqs = np.abs(np.fft.fftfreq(N_fft, d=1.0 / samplerate))
    # if regularization is None:
    #     eps_low = 1e-2
    #     eps_high = 1e-6
    #     f_low = 500.0
    #     f_high = 5000.0

    #     eps_array = np.zeros_like(freqs)
    #     log_eps_low = np.log10(eps_low)
    #     log_eps_high = np.log10(eps_high)

    #     for i, f in enumerate(freqs):
    #         if f < f_low:
    #             eps_array[i] = 10**log_eps_low
    #         elif f > f_high:
    #             eps_array[i] = 10**log_eps_high
    #         else:
    #             alpha = (np.log10(f + 1e-9) - np.log10(f_low)) / (np.log10(f_high) - np.log10(f_low))
    #             log_eps = (1 - alpha) * log_eps_low + alpha * log_eps_high
    #             eps_array[i] = 10**log_eps
    # else:
    #     raise ValueError("Regularization must be a callable or None.")
    
    # Step 4: Invert per frequency bin
    F11_freq = np.zeros(N_fft, dtype=np.complex128)
    F12_freq = np.zeros(N_fft, dtype=np.complex128)
    F21_freq = np.zeros(N_fft, dtype=np.complex128)
    F22_freq = np.zeros(N_fft, dtype=np.complex128)

    for k in range(N_fft):
        M = np.array([[h_ll[k], h_lr[k]],
                    [h_rl[k], h_rr[k]]])

        M_conjT = np.conj(M.T)
        M_reg = M_conjT @ M + eps_array[k] * np.eye(2)

        Minv = np.linalg.inv(M_reg) @ M_conjT

        F11_freq[k], F12_freq[k] = Minv[0, 0], Minv[0, 1]
        F21_freq[k], F22_freq[k] = Minv[1, 0], Minv[1, 1]

    # Step 5: IFFT to time domain
    f11_time = np.fft.ifft(F11_freq).real
    f12_time = np.fft.ifft(F12_freq).real
    f21_time = np.fft.ifft(F21_freq).real
    f22_time = np.fft.ifft(F22_freq).real

    def window_and_trim(ir, target_length):
        if len(ir) < target_length:
            ir = np.pad(ir, (0, target_length - len(ir)))
        elif len(ir) > target_length:
            ir = ir[:target_length]
        window = np.hanning(target_length)
        return ir * window

    f11_time = window_and_trim(f11_time, filter_length)
    f12_time = window_and_trim(f12_time, filter_length)
    f21_time = window_and_trim(f21_time, filter_length)
    f22_time = window_and_trim(f22_time, filter_length)

    # Frequency-domain inversion check: H(ω) @ F(ω)^T ≈ I for each frequency bin
    H_freq = np.array([[h_ll, h_lr], [h_rl, h_rr]])
    F_freq = np.array([[F11_freq, F12_freq], [F21_freq, F22_freq]])
    # Compute H * F^T over all frequency bins: HFt[i,j,k] = Σ_m H_freq[i,m,k] * F_freq[j,m,k]
    HFt = np.einsum('imk,jmk->ijk', H_freq, F_freq)
    # Broadcast identity matrix across frequency bins

    print("HFt shape: ", HFt.shape)
    identity = np.eye(2)[:, :, None]
    assert np.allclose(HFt, identity, atol=1e-6), "Frequency-domain inversion check failed"


    assert f11_time.shape == (filter_length,)
    assert f12_time.shape == (filter_length,)
    assert f21_time.shape == (filter_length,)
    assert f22_time.shape == (filter_length,)
    print("Passed shape assertions")

    
    return f11_time, f12_time, f21_time, f22_time

def generate_xtc_filters_mimo_rls(H_LL, H_LR, H_RL, H_RR, samplerate, head_position, speaker_positions, ear_offset=0.15, regularization=None, filter_length=4096):
    """
    Adaptive Least-Squares XTC filter design following Mouchtaris et al.

    Parameters match previous function for compatibility.
    """

    c = 343.0  # Speed of sound in m/s

    # Step 1: Align HRIRs
    H_LL, H_LR, H_RL, H_RR = align_impulse_responses(H_LL, H_LR, H_RL, H_RR)

    # Step 2: Zero-pad HRIRs
    def pad(ir, target_len):
        if len(ir) < target_len:
            return np.pad(ir, (0, target_len - len(ir)))
        else:
            return ir[:target_len]

    H_LL = pad(H_LL, filter_length)
    H_LR = pad(H_LR, filter_length)
    H_RL = pad(H_RL, filter_length)
    H_RR = pad(H_RR, filter_length)

    # Step 3: Build convolution matrices
    from scipy.linalg import toeplitz

    def build_toeplitz(signal, num_taps):
        col = np.concatenate(([signal[0]], np.zeros(num_taps-1)))
        row = signal[:num_taps]
        return toeplitz(col, row)

    T_LL = build_toeplitz(H_LL, filter_length)
    T_LR = build_toeplitz(H_LR, filter_length)
    T_RL = build_toeplitz(H_RL, filter_length)
    T_RR = build_toeplitz(H_RR, filter_length)

    # Step 4: Build system matrix A and desired response vector d
    A = np.block([
        [T_LL, T_LR, np.zeros_like(T_LL), np.zeros_like(T_LR)],
        [np.zeros_like(T_RL), np.zeros_like(T_RR), T_RL, T_RR]
    ])  # Shape: (2*filter_length, 4*filter_length)

    d = np.zeros(2*filter_length)
    d[:filter_length] = 1.0  # Left ear response: 1
    d[filter_length:] = 1.0  # Right ear response: 1

    # Step 5: Solve weighted least-squares problem
    lambda_reg = 1e-4
    ATA = A.T @ A + lambda_reg * np.eye(4*filter_length)
    ATd = A.T @ d
    filters = np.linalg.solve(ATA, ATd)

    # Step 6: Extract filters
    f11 = filters[:filter_length]
    f12 = filters[filter_length:2*filter_length]
    f21 = filters[2*filter_length:3*filter_length]
    f22 = filters[3*filter_length:]

    # Step 7: Window and output
    def window_and_trim(ir, target_len):
        if len(ir) < target_len:
            ir = np.pad(ir, (0, target_len - len(ir)))
        elif len(ir) > target_len:
            ir = ir[:target_len]
        window = np.hanning(target_len)
        return ir * window

    f11 = window_and_trim(f11, filter_length)
    f12 = window_and_trim(f12, filter_length)
    f21 = window_and_trim(f21, filter_length)
    f22 = window_and_trim(f22, filter_length)

    print(f"Max f11 (L->L): {np.max(np.abs(f11))}")
    print(f"Max f12 (L->R): {np.max(np.abs(f12))}")
    print(f"Max f21 (R->L): {np.max(np.abs(f21))}")
    print(f"Max f22 (R->R): {np.max(np.abs(f22))}")

    return f11, f12, f21, f22

def generate_xtc_filters_mimo_hybrid(
    H_LL, H_LR, H_RL, H_RR,
    samplerate, head_position, speaker_positions,
    ear_offset=0.15, regularization=None,
    filter_length=4096, num_iterations=10, fft_bins=1024
):    
    """
    Hybrid XTC filter design combining time-domain contralateral cancellation and
    frequency-domain flatness optimization.

    Steps:
    1. Align HRIRs using align_impulse_responses.
    2. Pad HRIRs to the target filter_length.
    3. Build convolution matrices and construct system matrix A and target vector d
       for time-domain contralateral cancellation.
    4. Solve the weighted least-squares problem to obtain initial filters.
    5. Perform FFT of the resulting filters.
    6. Define a flatness penalty that penalizes deviation of FFT magnitude from 1.0
       in ipsilateral filters (f11 and f22).
    7. Define a hybrid cost: time-domain contralateral energy + 0.001 * frequency-domain flatness error.
    8. Iteratively re-solve the least-squares problem (5 iterations) to improve frequency flatness.
    9. Extract f11, f12, f21, f22 and apply a Hanning window and trim to filter_length.
    10. Print debug information: maximum values of f11, f12, f21, f22.
    """
    import numpy as np
    from scipy.linalg import toeplitz
    print("Generating XTC filters using hybrid optimization...")
    c = 343.0  # speed of sound in m/s

    # Step 1: Align HRIRs
    H_LL, H_LR, H_RL, H_RR = align_impulse_responses(H_LL, H_LR, H_RL, H_RR)
    print("HRIRs aligned.")
    # Step 2: Pad HRIRs to target filter_length
    def pad(ir, target_len):
        if len(ir) < target_len:
            return np.pad(ir, (0, target_len - len(ir)))
        else:
            return ir[:target_len]

    H_LL = pad(H_LL, filter_length)
    H_LR = pad(H_LR, filter_length)
    H_RL = pad(H_RL, filter_length)
    H_RR = pad(H_RR, filter_length)

    # Step 3: Build convolution (Toeplitz) matrices for each HRIR
    def build_toeplitz(signal, num_taps):
        col = np.concatenate(([signal[0]], np.zeros(num_taps - 1)))
        row = signal[:num_taps]
        return toeplitz(col, row)

    print("Building convolution matrices...")
    T_LL = build_toeplitz(H_LL, filter_length)
    T_LR = build_toeplitz(H_LR, filter_length)
    T_RL = build_toeplitz(H_RL, filter_length)
    T_RR = build_toeplitz(H_RR, filter_length)

    print("Building system matrix A and desired response vector d...")
    # Step 4: Build system matrix A and desired response vector d
    A = np.block([
        [T_LL, T_LR, np.zeros_like(T_LL), np.zeros_like(T_LR)],
        [np.zeros_like(T_RL), np.zeros_like(T_RR), T_RL, T_RR]
    ])

    d = np.zeros(2 * filter_length)

    # Calculate correct impulse delays based on head position
    c = 343.0  # speed of sound
    L_ear = head_position + np.array([-ear_offset / 2, 0.0])
    R_ear = head_position + np.array([ear_offset / 2, 0.0])

    def compute_delay_samples(speaker_pos, ear_pos):
        distance = np.linalg.norm(speaker_pos - ear_pos)
        delay_sec = distance / c
        delay_samples = int(round(delay_sec * samplerate))
        return delay_samples

    delay_LL = compute_delay_samples(speaker_positions[0], L_ear)
    delay_RR = compute_delay_samples(speaker_positions[1], R_ear)

    if delay_LL < filter_length:
        d[delay_LL] = 1.0
    if delay_RR < filter_length:
        d[filter_length + delay_RR] = 1.0

    print("System matrix A and target vector d built.")
    print("Regularization for time-domain LS...")
    # Regularization for time-domain LS
    lambda_reg = 1e-4
    ATA = A.T @ A + lambda_reg * np.eye(4 * filter_length)
    ATd = A.T @ d
    AT = A.T
    errors = []
    # Step 5: Solve initial least-squares problem
    from scipy.linalg import cho_factor, cho_solve

    lambda_reg = 1e-4
    ATA = A.T @ A + lambda_reg * np.eye(4 * filter_length)
    ATd = A.T @ d

    # Precompute Cholesky
    c_factor = cho_factor(ATA)

    # Solve once initially
    filters = cho_solve(c_factor, ATd)
    # Extract initial filters
    f11 = filters[:filter_length]
    f12 = filters[filter_length:2 * filter_length]
    f21 = filters[2 * filter_length:3 * filter_length]
    f22 = filters[3 * filter_length:]

    # Step 6 & 7: Iterative re-optimization with hybrid cost (time-domain error + flatness penalty)
    weight_flatness = 0.001
    num_iterations = 5
    for iteration in range(num_iterations):
        # FFTs for ipsilateral filters
        F11 = np.fft.fft(f11, n=fft_bins)
        F22 = np.fft.fft(f22, n=fft_bins)

        # Flatness penalties
        flatness_error_F11 = np.abs(np.abs(F11) - 1.0)
        flatness_error_F22 = np.abs(np.abs(F22) - 1.0)
        flatness_penalty = weight_flatness * (np.mean(flatness_error_F11) + np.mean(flatness_error_F22))

        errors.append(flatness_penalty)

        # Update ATd slightly instead of d itself
        ATd_hybrid = ATd - flatness_penalty * np.ones_like(ATd)
        filters = cho_solve(c_factor, ATd_hybrid)

        # Update filters
        f11 = filters[:filter_length]
        f12 = filters[filter_length:2 * filter_length]
        f21 = filters[2 * filter_length:3 * filter_length]
        f22 = filters[3 * filter_length:]
        # Print progress bar
        bar_len = 30
        filled_len = int(bar_len * (iteration + 1) / num_iterations)
        bar = '#' * filled_len + '-' * (bar_len - filled_len)
        print(f"\r[{bar}] {100*(iteration+1)//num_iterations}% Iter {iteration+1}/{num_iterations} | Error: {flatness_penalty:.6f}", end='')
    print("\nOptimization complete.")
    # Step 8: Apply Hanning window and trim to filter_length
    def window_and_trim(ir, target_len):
        if len(ir) < target_len:
            ir = np.pad(ir, (0, target_len - len(ir)))
        elif len(ir) > target_len:
            ir = ir[:target_len]
        window = np.hanning(target_len)
        return ir * window

    f11 = window_and_trim(f11, filter_length)
    f12 = window_and_trim(f12, filter_length)
    f21 = window_and_trim(f21, filter_length)
    f22 = window_and_trim(f22, filter_length)

    # Step 9: Debug prints for filter maxima
    print(f"Hybrid Optimization Debug: Max f11: {np.max(np.abs(f11))}")
    print(f"Hybrid Optimization Debug: Max f12: {np.max(np.abs(f12))}")
    print(f"Hybrid Optimization Debug: Max f21: {np.max(np.abs(f21))}")
    print(f"Hybrid Optimization Debug: Max f22: {np.max(np.abs(f22))}")

    return f11, f12, f21, f22

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
            h_ll, h_lr = extract_hrirs_sam(sofa_file, -30, True)
            h_rl, h_rr = extract_hrirs_sam(sofa_file, 30, True)
        elif choice == "4":
            print("Entering record mode...")
            recordHRTF()
            break
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")



if __name__ == "__main__":
    main()

