


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_saved_xtc_data(results_folder="results/xtc_dataclose", samplerate=48000):
    def load_csv(name):
        return np.loadtxt(os.path.join(results_folder, name), delimiter=",")

    # Time-domain HRIRs
    fig1, axs1 = plt.subplots(2, 2, figsize=(10, 6))
    axs1[0, 0].plot(load_csv("H_LL_raw_ir.csv"))
    axs1[0, 0].set_title("H_LL raw IR")
    axs1[0, 1].plot(load_csv("H_LR_raw_ir.csv"))
    axs1[0, 1].set_title("H_LR raw IR")
    axs1[1, 0].plot(load_csv("H_RL_raw_ir.csv"))
    axs1[1, 0].set_title("H_RL raw IR")
    axs1[1, 1].plot(load_csv("H_RR_raw_ir.csv"))
    axs1[1, 1].set_title("H_RR raw IR")
    fig1.suptitle("Original HRIRs (Time Domain)")
    fig1.tight_layout()

    # Filter IRs
    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 6))
    axs2[0, 0].plot(load_csv("fLL_ir.csv"))
    axs2[0, 0].set_title("fLL IR")
    axs2[0, 1].plot(load_csv("fLR_ir.csv"))
    axs2[0, 1].set_title("fLR IR")
    axs2[1, 0].plot(load_csv("fRL_ir.csv"))
    axs2[1, 0].set_title("fRL IR")
    axs2[1, 1].plot(load_csv("fRR_ir.csv"))
    axs2[1, 1].set_title("fRR IR")
    fig2.suptitle("XTC Filters (Time Domain)")
    fig2.tight_layout()

    # HF Product FFTs
    # For frequencies, estimate N from X_LL_fft.csv: shape[0] = rfft size, so time-domain length is (n-1)*2+1
    n_rfft = load_csv("X_LL_fft.csv").shape[0]
    N = (n_rfft - 1) * 2
    freqs = np.fft.rfftfreq(N, d=1 / samplerate)
    fig3, axs3 = plt.subplots(2, 2, figsize=(10, 6))
    axs3[0, 0].plot(freqs, 20 * np.log10(load_csv("X_LL_fft.csv") + 1e-9))
    axs3[0, 0].set_title("X_LL FFT (HF product)")
    axs3[0, 1].plot(freqs, 20 * np.log10(load_csv("X_LR_fft.csv") + 1e-9))
    axs3[0, 1].set_title("X_LR FFT")
    axs3[1, 0].plot(freqs, 20 * np.log10(load_csv("X_RL_fft.csv") + 1e-9))
    axs3[1, 0].set_title("X_RL FFT")
    axs3[1, 1].plot(freqs, 20 * np.log10(load_csv("X_RR_fft.csv") + 1e-9))
    axs3[1, 1].set_title("X_RR FFT")
    fig3.suptitle("HF Product Matrix (dB)")
    fig3.tight_layout()

    # Comparison plots for crosstalk cancellation
    def plot_fft_db(signal):
        spectrum = np.fft.rfft(signal)
        freqs_local = np.fft.rfftfreq(len(signal), d=1 / samplerate)
        return freqs_local, 20 * np.log10(np.abs(spectrum) + 1e-9)

    fig4, axs4 = plt.subplots(2, 1, figsize=(10, 8))

    freqs_ll, mag_ll = plot_fft_db(load_csv("M_LL.csv"))
    freqs_rl, mag_rl = plot_fft_db(load_csv("M_RL.csv"))
    axs4[0].plot(freqs_ll, mag_ll, label="M_LL")
    axs4[0].plot(freqs_rl, mag_rl, label="M_RL")
    axs4[0].set_title("Left Input to Both Ears (FFT)")
    axs4[0].set_xlim(0, 12000)
    axs4[0].set_ylim(-10, 50)
    axs4[0].legend()
    axs4[0].grid(True)

    freqs_rr, mag_rr = plot_fft_db(load_csv("M_RR.csv"))
    freqs_lr, mag_lr = plot_fft_db(load_csv("M_LR.csv"))
    axs4[1].plot(freqs_rr, mag_rr, label="M_RR")
    axs4[1].plot(freqs_lr, mag_lr, label="M_LR")
    axs4[1].set_title("Right Input to Both Ears (FFT)")
    axs4[1].set_xlim(0, 12000)
    axs4[1].set_ylim(-10, 50)
    axs4[1].legend()
    axs4[1].grid(True)

    fig4.suptitle("Measured Crosstalk Cancellation Comparison (FFT dB)")
    fig4.tight_layout()

    plt.show()

plot_saved_xtc_data()