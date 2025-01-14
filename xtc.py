import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import pysofaconventions as sofa
import os
from scipy.fft import fft, ifft, fftfreq

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
    return {
        "H_LL": H_LL,  # shape (L,)
        "H_LR": H_LR,
        "H_RL": H_RL,
        "H_RR": H_RR,
        "samplerate": samplerate
    }


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
    return {
        "H_LL": H_LL,  # shape (L,)
        "H_LR": H_LR,
        "H_RL": H_RL,
        "H_RR": H_RR,
        "samplerate": samplerate
    }


def generate_xtc_filters_mimo(H_LL, H_LR, H_RL, H_RR, samplerate, regularization=0.01):
    """
    Build 2x2 crosstalk canceller using the 4 IRs:
      H_LL(t) = speaker L -> ear L
      H_LR(t) = speaker L -> ear R
      H_RL(t) = speaker R -> ear L
      H_RR(t) = speaker R -> ear R

    Returns 4 time-domain filters:
      (F11, F12, F21, F22)
    so that:
      speaker_L(t) = F11 * ear_L(t) + F12 * ear_R(t)
      speaker_R(t) = F21 * ear_L(t) + F22 * ear_R(t)
    """
    import numpy as np
    from scipy.fft import fft, ifft

    # 1) Zero-pad to consistent length
    L = max(len(H_LL), len(H_LR), len(H_RL), len(H_RR))
    N_fft = 2**int(np.ceil(np.log2(L)))
    h_ll = np.fft.fft(H_LL, n=N_fft)
    h_lr = np.fft.fft(H_LR, n=N_fft)
    h_rl = np.fft.fft(H_RL, n=N_fft)
    h_rr = np.fft.fft(H_RR, n=N_fft)

    # 2) invert each frequency bin of the 2x2 matrix:
    #    M = [[h_ll, h_lr],
    #         [h_rl, h_rr]]
    # M^-1 = 1/det(M) * [[h_rr, -h_lr], [-h_rl, h_ll]]
    # with small regularization 'eps' on the diagonal or in det(M).
    eps = regularization
    F11_freq = np.zeros(N_fft, dtype=np.complex128)
    F12_freq = np.zeros(N_fft, dtype=np.complex128)
    F21_freq = np.zeros(N_fft, dtype=np.complex128)
    F22_freq = np.zeros(N_fft, dtype=np.complex128)

    for k in range(N_fft):
        M11, M12 = h_ll[k], h_lr[k]
        M21, M22 = h_rl[k], h_rr[k]
        det = (M11*M22 - M12*M21) + eps
        inv11 =  M22 / det
        inv12 = -M12 / det
        inv21 = -M21 / det
        inv22 =  M11 / det

        # So if input=[Ear_L, Ear_R], output=[Spk_L, Spk_R],
        # then:
        # F11=inv11, F12=inv12, F21=inv21, F22=inv22
        F11_freq[k] = inv11
        F12_freq[k] = inv12
        F21_freq[k] = inv21
        F22_freq[k] = inv22

    # 3) iFFT to get time-domain
    f11_time = np.fft.ifft(F11_freq).real[:L]
    f12_time = np.fft.ifft(F12_freq).real[:L]
    f21_time = np.fft.ifft(F21_freq).real[:L]
    f22_time = np.fft.ifft(F22_freq).real[:L]

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
        print("4. Exit")

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
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()