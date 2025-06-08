import numpy as np
from math import sin, cos, radians
from tqdm import tqdm
import os
from scipy.optimize import minimize
from scipy.interpolate import interp1d
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
    plt.title("IR Before Alignment")
    plt.plot(np.arange(len(H_LL)), H_LL, 'o', label="IR")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.title("IR After Alignment")
    plt.plot(np.arange(len(H_LL_new)), H_LL_new, 'o', label="IR_Rolled")
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

def extract_hrirs_sam(filepath, source_az, show_plots = False, attempt_interpolate=False):
    import os
    cache_file = "hrir_cache_lookup.npz"
    if not os.path.exists(cache_file):
        print("Generating HRIR cache...")
        hrtf = spatialaudiometrics.load_data.HRTF(filepath)
        cache_dict = {}
        for az in range(0, 360, 5):
            idx = np.where((hrtf.locs[:, 0] == az) & (hrtf.locs[:, 1] == 0))[0]
            if len(idx) > 0:
                idx = idx[0]
                hrir_l = hrtf.hrir[idx, 0, :]
                hrir_r = hrtf.hrir[idx, 1, :]
                cache_dict[f"{az}"] = (hrir_l, hrir_r)
        np.savez(cache_file, **cache_dict)
        print("HRIR cache saved.")

    cache = np.load(cache_file, allow_pickle=True)
    samplerate = sofa.SOFAFile(filepath, 'r').getSamplingRate()

    source_az = source_az % 360
    if not attempt_interpolate:
        key = f"{int(round(source_az / 5) * 5) % 360}"
        hrir_l, hrir_r = cache[key]
    else:
        floor_az = int(np.floor(source_az / 5) * 5) % 360
        ceil_az = int(np.ceil(source_az / 5) * 5) % 360
        if floor_az == ceil_az:
            hrir_l, hrir_r = cache[f"{floor_az}"]
        else:
            hrir_l_floor, hrir_r_floor = cache[f"{floor_az}"]
            hrir_l_ceil, hrir_r_ceil = cache[f"{ceil_az}"]
            # Handle wrapping for interp_factor
            az_diff = (ceil_az - floor_az) % 360
            if az_diff == 0:
                az_diff = 5
            interp_factor = (source_az - floor_az) / az_diff
            hrir_l = np.fft.ifft((1 - interp_factor) * np.fft.fft(hrir_l_floor) + interp_factor * np.fft.fft(hrir_l_ceil)).real
            hrir_r = np.fft.ifft((1 - interp_factor) * np.fft.fft(hrir_r_floor) + interp_factor * np.fft.fft(hrir_r_ceil)).real

    # if(show_plots):
        # If you want to visualize, need to load HRTF for plotting
        # hrtf = spatialaudiometrics.load_data.HRTF(filepath)
        # fig,gs = spatialaudiometrics.visualisation.create_fig()
        # axes = fig.add_subplot(gs[1:6,1:6])
        # spatialaudiometrics.visualisation.plot_hrir_both_ears(hrtf,source_az,0,axes)
        # axes = fig.add_subplot(gs[1:6,7:12])
        # spatialaudiometrics.visualisation.plot_itd_overview(hrtf)
        # spatialaudiometrics.visualisation.show()
        # plt.show()
    return hrir_l, hrir_r, samplerate


def extract_hrirs_sam_uncache(filepath, source_az, show_plots = False, attempt_interpolate=False):
    # If we're not attempting to interpolate, round to the nearest 5 degrees
    print("attempting extraction")
    hrtf = spatialaudiometrics.load_data.HRTF(filepath)
    if source_az < 0:
        source_az += 360

    if not attempt_interpolate:
        source_az = round(source_az / 5) * 5

        idx = np.where((hrtf.locs[:,0] == source_az) & (hrtf.locs[:,1] == 0))[0][0]

        hrir_l = hrtf.hrir[idx,0,:]
        hrir_r = hrtf.hrir[idx,1,:]
    
    else:
        # Attempt to interpolate if exact azimuth not found
        # find bracketed azimuths
        print(f"Interpolating HRIR for azimuth {source_az} degrees")
        source_az_floor = np.floor(source_az / 5) * 5
        source_az_ceil = np.ceil(source_az / 5) * 5
        # Wrap around 360° to 0°
        source_az_floor = source_az_floor % 360
        source_az_ceil = source_az_ceil % 360
        if source_az_floor == source_az_ceil:
            idx = np.where((hrtf.locs[:,0] == source_az_floor) & (hrtf.locs[:,1] == 0))[0][0]
            hrir_l = hrtf.hrir[idx,0,:]
            hrir_r = hrtf.hrir[idx,1,:]


            samplerate = sofa.SOFAFile(filepath, 'r').getSamplingRate()
            return hrir_l, hrir_r, samplerate
        print(f"Floor: {source_az_floor}, Ceil: {source_az_ceil}")
        # grab both HRIRs
        idx_floor = np.where((hrtf.locs[:,0] == source_az_floor) & (hrtf.locs[:,1] == 0))[0][0]
        idx_ceil = np.where((hrtf.locs[:,0] == source_az_ceil) & (hrtf.locs[:,1] == 0))[0][0]
        hrir_l_floor = hrtf.hrir[idx_floor,0,:]
        hrir_r_floor = hrtf.hrir[idx_floor,1,:]
        hrir_l_ceil = hrtf.hrir[idx_ceil,0,:]
        hrir_r_ceil = hrtf.hrir[idx_ceil,1,:]
        # take FFT for all four
        hrir_l_floor_fft = np.fft.fft(hrir_l_floor)
        hrir_r_floor_fft = np.fft.fft(hrir_r_floor)
        hrir_l_ceil_fft = np.fft.fft(hrir_l_ceil)
        hrir_r_ceil_fft = np.fft.fft(hrir_r_ceil)
        # interpolate L in frequency domain, acounting for position of source_az between floor and ceil
        interp_factor = (source_az - source_az_floor) / ((source_az_ceil - source_az_floor) % 360 if (source_az_ceil - source_az_floor) % 360 != 0 else 5)
        hrir_l_fft = (1 - interp_factor) * hrir_l_floor_fft + interp_factor * hrir_l_ceil_fft
        hrir_r_fft = (1 - interp_factor) * hrir_r_floor_fft + interp_factor * hrir_r_ceil_fft
        # take IFFT to get time domain HRIRs
        hrir_l = np.fft.ifft(hrir_l_fft).real
        hrir_r = np.fft.ifft(hrir_r_fft).real

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

def generate_filter(h_LL, h_LR, h_RL, h_RR, speaker_positions, head_position, head_angle=None, filter_length=2048, samplerate=48000, debug=False, format='TimeDomain'):
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
    from scipy.signal import firwin
    # Design FIR high-pass filter (match block processing)
    numtaps_hp = 1025  # same as audio engine originally used
    h_hp = firwin(numtaps_hp, cutoff=80.0, fs=samplerate, pass_zero=False)
    # Zero-pad to length N
    h_hp_padded = np.pad(h_hp, (0, N - numtaps_hp))
    # FFT to get frequency response
    H_hp = np.fft.fft(h_hp_padded, n=N)
    # Multiply each filter's frequency response by the HP response
    FLL = FLL * H_hp
    FLR = FLR * H_hp
    FRL = FRL * H_hp
    FRR = FRR * H_hp
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

def generate_kn_filter(h_LL, h_LR, h_RL, h_RR, speaker_positions, head_position, head_angle=None, filter_length=2048, samplerate=48000, debug=False, lambda_freq=1e-4, format='FrequencyDomain'):

    # Initialize cache if not already initialized
    if filter_cache is None or cache_stats is None:
        init_filter_cache()
    # --- Memoization with best-match tolerance-based lookup using head position and angle ---
    def get_azimuth(p1, p2):
        delta = np.array(p2) - np.array(p1)
        return np.degrees(np.arctan2(delta[1], delta[0])) % 360

    az_L = get_azimuth(head_position, speaker_positions[0])
    az_R = get_azimuth(head_position, speaker_positions[1])
    key = (az_L, az_R, head_angle, head_position[0], head_position[1])

    cache_stats["TotalCalls"] += 1

    best_match = None
    min_distance = float("inf")
    for cached_key in filter_cache:
        dL = abs((cached_key[0] - az_L + 180) % 360 - 180)
        dR = abs((cached_key[1] - az_R + 180) % 360 - 180)
        dA = abs((cached_key[2] - head_angle + 180) % 360 - 180)
        dX = abs(cached_key[3] - head_position[0])
        dY = abs(cached_key[4] - head_position[1])
        distance = max(dL, dR, dA, dX, dY)
        # print(f"[DEBUG] Comparing to cached key: L = {cached_key[0]:.2f}, R = {cached_key[1]:.2f}, A = {cached_key[2]:.2f}, X = {cached_key[3]:.2f}, Y = {cached_key[4]:.2f}")
        # print(f"[DEBUG] dL = {dL:.2f}, dR = {dR:.2f}, dA = {dA:.2f}, dX = {dX:.2f}, dY = {dY:.2f}")
        if distance < min_distance:
            min_distance = distance
            best_match = cached_key

    if best_match is not None and min_distance <= cache_tolerance:
        # print(f"[DEBUG] Cache hit: Using cached filter for key {best_match}")
        cache_stats["CacheHits"] += 1
        hit_rate = cache_stats["CacheHits"] / cache_stats["TotalCalls"]
        with open(cache_stats_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([cache_stats["TotalCalls"], cache_stats["CacheHits"], f"{hit_rate:.4f}"])
        return filter_cache[best_match]
    else:
        print(f"[DEBUG] Cache miss or outside tolerance (min_distance = {min_distance:.2f}); generating new filter.")

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

    # Store in cache and log stats
    filter_cache[key] = (FLL, FLR, FRL, FRR)
    hit_rate = cache_stats["CacheHits"] / cache_stats["TotalCalls"]
    with open(cache_stats_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([cache_stats["TotalCalls"], cache_stats["CacheHits"], f"{hit_rate:.4f}"])

    from scipy.signal import firwin
    # Design FIR high-pass filter (match block processing)
    numtaps_hp = 1025  # same as audio engine originally used
    h_hp = firwin(numtaps_hp, cutoff=80.0, fs=samplerate, pass_zero=False)
    # Zero-pad to length N
    h_hp_padded = np.pad(h_hp, (0, N - numtaps_hp))
    # FFT to get frequency response
    H_hp = np.fft.fft(h_hp_padded, n=N)
    # Multiply each filter's frequency response by the HP response
    FLL = FLL * H_hp
    FLR = FLR * H_hp
    FRL = FRL * H_hp
    FRR = FRR * H_hp
    return FLL, FLR, FRL, FRR  # fll, flr, frl, frr

def generate_real_kn_filter(
    h_LL, h_LR, h_RL, h_RR,
    filter_length=16384, samplerate=48000, debug=False, lambda_freq=1e-4, format='FrequencyDomain'
):
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
    freqs = np.fft.fftfreq(N, d=1/samplerate)
    def lambda_fn(k, N):
        f = abs(freqs[k])
        if f < 8000:
            return 1e-4
        elif f < 16000:
            return 1e-3
        else:
            return lambda_freq * (1 + ((f - 8000) / 4000) **2)
    # Step 3: Frequency domain Kirkeby-Nelson inversion
    F_f = np.zeros((2, 2, N), dtype=complex)
    for k in range(N):
        Hk = H_f[:, :, k]
        Hk_H = Hk.conj().T
        lambda_k = lambda_fn(k, N)
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

    # Store in cache and log stats
    return FLL, FLR, FRL, FRR  # fll, flr, frl, frr


def generate_real_kn_filter_fd(
    H_LL, H_LR, H_RL, H_RR,
    filter_length=16384, samplerate=48000, debug=False, lambda_freq=1e-4, format='FrequencyDomain'
):
    from scipy.fft import fft, fftfreq
    filter_length = len(H_LL)  # Use the length of the provided HRIRs
    N = filter_length  # Use the provided filter length directly
    H_f = np.array([[H_LL, H_LR], [H_RL, H_RR]])
    
    # Step 3: Frequency domain Kirkeby-Nelson inversion
    F_f = np.zeros((2, 2, N), dtype=complex)
    # set lambda freq that supresses HF ringing
    freqs = np.fft.fftfreq(N, d=1/samplerate)
    def lambda_fn(k, N):
        f = abs(freqs[k])
        if f < 4000:
            return 1e-3
        elif f < 8000:
            return 1e-2
        elif f < 12000:
            return 1e-1
        else:
            return lambda_freq * (1 + ((f - 8000) / 4000) **2)

    for k in range(N):
        Hk = H_f[:, :, k]
        Hk_H = Hk.conj().T
        lambda_k = lambda_fn(k, N)
        inv = np.linalg.inv(Hk_H @ Hk + lambda_k * np.eye(2)) @ Hk_H
        F_f[:, :, k] = inv  # targeting D = identity matrix

    FLL = F_f[0, 0, :]
    FLR = F_f[0, 1, :]
    FRL = F_f[1, 0, :]
    FRR = F_f[1, 1, :]
    # multiply by 700 sample delay.
    delay_samples = -700
    omega = 2 * np.pi * fftfreq(N, d=1/samplerate)
    tau_L = delay_samples / samplerate
    tau_R = delay_samples / samplerate
    exp_L = np.exp(1j * omega * tau_L)    # for the left speaker
    exp_R = np.exp(1j * omega * tau_R)    # for the right speaker
    # FLL *= exp_L
    # FLR *= exp_R
    # FRL *= exp_L
    # FRR *= exp_R
    # bandpass filter
    bpfilter = np.zeros(N, dtype=complex)
    freqs = np.fft.fftfreq(N, d=1/samplerate)
    for i in range(N):
        if 120 <= abs(freqs[i]) <= 10000:
            bpfilter[i] = 1.0
            if 10000< abs(freqs[i]) < 16000:
                bpfilter[i] *= 0.8
        # elif 10000 < abs(freqs[i]) < 16000:
        #     bpfilter[i] = 0
        else:
            bpfilter[i] = 0.0
    
    FLL *= bpfilter
    FLR *= bpfilter
    FRL *= bpfilter
    FRR *= bpfilter

    # Store in cache and log stats
    return FLL, FLR, FRL, FRR  # fll, flr, frl, frr

def generate_real_kn_filter_fd_smart(
    H_LL, H_LR, H_RL, H_RR,
    filter_length=16384, samplerate=48000, debug=False, lambda_freq=1e-4, format='FrequencyDomain'
):
    from scipy.fft import fft, fftfreq
    c = 343.0
    filter_length = len(H_LL)  # Use the length of the provided HRIRs
    N = filter_length  # Use the provided filter length directly
    H_f = np.array([[H_LL, H_LR], [H_RL, H_RR]])


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
        log_lambda_w[safe_mask] = lambda_interp_func(np.log10(freqs[safe_mask]))
        lambda_w = 10 ** log_lambda_w
        # For negative/zero freqs, fallback to first positive value
        if np.any(safe_mask):
            lambda_w[~safe_mask] = lambda_w[safe_mask][0]
        else:
            lambda_w[:] = 1e-3  # fallback if no positive freqs (shouldn't happen)

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
    print("starting optimisation")
    res = minimize(
        compute_regularization_error,
        init_log_lambda,
        args=(H_f, freqs, freq_mask),
        method='L-BFGS-B',
        bounds=[(-8, 2)] * 8,
        options={"maxiter": 25}
    )

    print("finished optimisation")


    optimized_log_lambda = res.x
    positive_freqs = freqs[freqs > 0]
    control_freqs = np.geomspace(positive_freqs[0], positive_freqs[-1], len(optimized_log_lambda))
    lambda_interp_func = interp1d(np.log10(control_freqs), optimized_log_lambda, kind='linear', fill_value="extrapolate")
    # Safe interpolation: only for positive frequencies
    log_lambda_w = np.full_like(freqs, fill_value=np.nan, dtype=float)
    safe_mask = freqs > 0
    log_lambda_w[safe_mask] = lambda_interp_func(np.log10(freqs[safe_mask]))
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
    bpfilter = np.zeros(N, dtype=complex)
    freqs = np.fft.fftfreq(N, d=1/samplerate)
    for i in range(N):
        if 200 <= abs(freqs[i]) <= 10000:
            bpfilter[i] = 1.0
        else:
            bpfilter[i] = 0.0
    
    FLL *= bpfilter
    FLR *= bpfilter
    FRL *= bpfilter
    FRR *= bpfilter


    # Store in cache and log stats
    return FLL, FLR, FRL, FRR  # fll, flr, frl, frr



def generate_kn_filter_smart(
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
        log_lambda_w[safe_mask] = lambda_interp_func(np.log10(freqs[safe_mask]))
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
    log_lambda_w[safe_mask] = lambda_interp_func(np.log10(freqs[safe_mask]))
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

def generate_real_kn_filter_smart(
    h_LL, h_LR, h_RL, h_RR,
    filter_length=2048, samplerate=48000, debug=False, lambda_freq=1e-4, format='FrequencyDomain'
):
    print("Generating real Kirkeby-Nelson filter with smart optimization...")
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
        log_lambda_w[safe_mask] = lambda_interp_func(np.log10(freqs[safe_mask]))
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
    print("Optimizing regularization parameters...")
    res = minimize(
        compute_regularization_error,
        init_log_lambda,
        args=(H_f, freqs, freq_mask),
        method='L-BFGS-B',
        bounds=[(-8, 2)] * 8,
        options={"maxiter": 25}
    )
    print("Optimization complete.")

    optimized_log_lambda = res.x
    positive_freqs = freqs[freqs > 0]
    control_freqs = np.geomspace(positive_freqs[0], positive_freqs[-1], len(optimized_log_lambda))
    lambda_interp_func = interp1d(np.log10(control_freqs), optimized_log_lambda, kind='linear', fill_value="extrapolate")
    # Safe interpolation: only for positive frequencies
    log_lambda_w = np.full_like(freqs, fill_value=np.nan, dtype=float)
    safe_mask = freqs > 0
    log_lambda_w[safe_mask] = lambda_interp_func(np.log10(freqs[safe_mask]))
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
    return FLL, FLR, FRL, FRR  # fll, flr, frl, frr



def plot_filter_surface(filter_coeffs, samplerate, title="Filter Surface"):
    import numpy as np
    import matplotlib.pyplot as plt

    # Pole-zero plot on the z-plane for an FIR filter
    zeros = np.roots(filter_coeffs)
    # FIR filters have no finite poles; all poles are at the origin
    poles = np.zeros(len(filter_coeffs) - 1)

    fig, ax = plt.subplots()
    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')
    # Plot zeros
    ax.plot(np.real(zeros), np.imag(zeros), 'o', label='Zeros')
    # Plot poles at origin
    if poles.size > 0:
        ax.plot(np.real(poles), np.imag(poles), 'x', label='Poles')
    ax.set_title(title)
    ax.set_xlabel('Real(z)')
    ax.set_ylabel('Imag(z)')
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.grid(True)
    plt.show()

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

def compute_rt60(ir, samplerate):
    """Estimate RT60 using the Schroeder backward integration method."""
    if ir is None or len(ir) == 0:
        return None

    energy = ir ** 2
    energy = energy / np.max(energy)  # Normalize

    sch = np.cumsum(energy[::-1])[::-1]
    sch_db = 10 * np.log10(sch + 1e-12)

    print("Schroeder dB range:", sch_db.min(), sch_db.max())  # Debug output

    # Relaxed thresholds to accommodate shallow decays
    start_idx = np.argmax(sch_db < -2)
    end_idx = np.argmax(sch_db < -20)

    if end_idx <= start_idx or end_idx == 0:
        return None

    t = np.arange(len(sch_db)) / samplerate
    slope, intercept = np.polyfit(t[start_idx:end_idx], sch_db[start_idx:end_idx], 1)

    rt60 = -60 / slope if slope < 0 else None
    return rt60

def fft_noise_subtraction(signal, noise, floor_db=-120):
    """
    Subtracts a noise spectrum from a signal in the frequency domain,
    with robust protection against underflow, overflow, and invalid values.
    """
    N = max(len(signal), len(noise))
    sig_fft = np.fft.fft(signal, n=N)
    noise_fft = np.fft.fft(noise, n=N)

    sig_mag = np.abs(sig_fft)
    noise_mag = np.abs(noise_fft)

    # Safe noise floor epsilon in linear magnitude
    try:
        eps = 10 ** (floor_db / 20)
        if not np.isfinite(eps) or eps < 1e-12:
            eps = 1e-12
    except OverflowError:
        eps = 1e-12

    # Subtract noise spectrum magnitude
    sub_mag = sig_mag - noise_mag
    sub_mag = np.clip(sub_mag, eps, np.max(sig_mag))

    # Preserve original phase
    sig_phase = np.angle(sig_fft)

    # Reconstruct spectrum
    sub_fft = sub_mag * np.exp(1j * sig_phase)

    # Return real part of inverse FFT
    return np.fft.ifft(sub_fft).real[:len(signal)]

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


# --- Helper function to generate cache statistics plot ---

def generate_cache_stats_plot():
    import matplotlib.pyplot as plt
    import csv
    import os
    global cache_stats_file
    if not os.path.exists(cache_stats_file):
        print("[DEBUG] No cache stats file found.")
        return

    total_calls = []
    hit_rates = []

    with open(cache_stats_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_calls.append(int(row["TotalCalls"]))
            hit_rates.append(float(row["HitRate"]))

    if total_calls and hit_rates:
        plt.figure(figsize=(8, 4))
        plt.plot(total_calls, hit_rates, marker='o')
        plt.xlabel("Total Calls")
        plt.ylabel("Cache Hit Rate")
        plt.title("Filter Cache Hit Rate Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("filter_cache_stats.png")
        print("[DEBUG] Cache statistics plot saved as filter_cache_stats.png")



def get_prerendered_filter(angle):
    """
    Load the cached filters and interpolate between adjacent angles if needed.
    Returns (FLL, FLR, FRL, FRR) for the requested angle (can be fractional).
    """
    from scipy.interpolate import interp1d

    if not os.path.exists("prerendered_filters.npz"):
        raise FileNotFoundError("prerendered_filters.npz not found. Run prerender_filters() first.")

    data = np.load("prerendered_filters.npz", allow_pickle=True)
    angle_floor = int(np.floor(angle)) % 360
    angle_ceil = int(np.ceil(angle)) % 360
    weight = angle - angle_floor

    def load_components(a):
        f = data[str(a)].item()
        return f['FLL'], f['FLR'], f['FRL'], f['FRR']

    FLL0, FLR0, FRL0, FRR0 = load_components(angle_floor)
    FLL1, FLR1, FRL1, FRR1 = load_components(angle_ceil)

    FLL = (1 - weight) * FLL0 + weight * FLL1
    FLR = (1 - weight) * FLR0 + weight * FLR1
    FRL = (1 - weight) * FRL0 + weight * FRL1
    FRR = (1 - weight) * FRR0 + weight * FRR1

    return FLL, FLR, FRL, FRR