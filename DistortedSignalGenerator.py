import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pywt
import glob
import warnings
from hyperopt import fmin, tpe, hp, Trials

# Suppress specific warning
warnings.filterwarnings("ignore",
                        message="Level value of .* is too high: all coefficients will experience boundary effects.",
                        module="pywt")


# Function to generate a signal with specified parameters
def generate_signal(num_samples, base_freq, sample_rate, amplitude, harmonic3, harmonic5, harmonic7, harmonic9,
                    harmonic11, transient_amount, transient_max_value):
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    base_signal = amplitude * np.sin(2 * np.pi * base_freq * t)
    harmonic_signal = (harmonic3 * amplitude * np.sin(2 * np.pi * 3 * base_freq * t) +
                       harmonic5 * amplitude * np.sin(2 * np.pi * 5 * base_freq * t) +
                       harmonic7 * amplitude * np.sin(2 * np.pi * 7 * base_freq * t) +
                       harmonic9 * amplitude * np.sin(2 * np.pi * 9 * base_freq * t) +
                       harmonic11 * amplitude * np.sin(2 * np.pi * 11 * base_freq * t))
    signal = base_signal + harmonic_signal
    transient_indices = np.random.choice(num_samples, transient_amount, replace=False)
    for idx in transient_indices:
        signal[idx] += transient_max_value * np.random.uniform(-1, 1)
    return t, signal


# Function to save signal data to a CSV file
def save_to_csv(filename, t, signal):
    df = pd.DataFrame({'Time': t, 'Signal': signal})
    df.to_csv(filename, index=False)


# Function to plot signals from CSV files
def plot_signals(pattern):
    # Find files matching the pattern
    filenames = glob.glob(pattern)

    if not filenames:
        print(f"No files matching pattern {pattern} were found.")
        return

    # Sort filenames for consistent plotting
    filenames.sort()

    # Extract indices for the first, middle, and last files
    num_files = len(filenames)
    first_idx = 0
    middle_idx = num_files // 2
    last_idx = num_files - 1

    # Select filenames for plotting
    filenames_to_plot = [filenames[first_idx], filenames[middle_idx], filenames[last_idx]]

    plt.figure(figsize=(15, 5))
    for i, filename in enumerate(filenames_to_plot):
        data = pd.read_csv(filename)
        t = data['Time']
        signal = data['Signal']
        plt.subplot(1, 3, i + 1)
        plt.plot(t, signal)
        plt.title(f'Signal from {os.path.basename(filename)}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


# Function to perform DWT and thresholding
def dwt_compress(signal, wavelet, level, threshold):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    thresholded_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return thresholded_coeffs


# Function to reconstruct signal from DWT coefficients
def dwt_reconstruct(coeffs, wavelet):
    return pywt.waverec(coeffs, wavelet)


# Function to compress and evaluate signal
def compress_and_evaluate(params, signal, mse_threshold):
    wavelet = params['wavelet']
    level = int(params['level'])
    threshold = params['threshold']
    coeffs = dwt_compress(signal, wavelet, level, threshold)
    compressed_signal = dwt_reconstruct(coeffs, wavelet)
    mse = np.mean((signal - compressed_signal[:len(signal)]) ** 2)
    if mse > mse_threshold:
        return 0  # Return 0 compression ratio if MSE is too high
    original_size = len(signal)
    compressed_size = sum(len(c) for c in coeffs)
    compression_ratio = original_size / compressed_size
    return compression_ratio


# Bayesian optimization to find optimal parameters
def optimize_parameters(signal, mse_threshold=0.01):
    wavelets = pywt.wavelist(kind='discrete')

    def objective(params):
        params['wavelet'] = wavelets[int(params['wavelet'])]
        compression_ratio = compress_and_evaluate(params, signal, mse_threshold)
        return -compression_ratio  # We minimize the negative compression ratio

    space = {
        'wavelet': hp.quniform('wavelet', 0, len(wavelets) - 1, 1),
        'level': hp.quniform('level', 1, 10, 1),
        'threshold': hp.uniform('threshold', 0.01, 1.0)
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    best['wavelet'] = wavelets[int(best['wavelet'])]
    best['level'] = int(best['level'])
    return best


# Main function to process signals and find optimal parameters
def main():
    num_files = 1000  # Number of signal files to generate
    num_samples = 1000  # Number of samples per signal
    base_freq = 50  # Base frequency in Hz
    sample_rate = 5000  # Sample rate in Hz
    amplitude = 230  # Amplitude of the signal

    # Define min and max values for harmonics and transient parameters
    harmonic3_min, harmonic3_max = 0, 0.3
    harmonic5_min, harmonic5_max = 0, 0.3
    harmonic7_min, harmonic7_max = 0, 0.3
    harmonic9_min, harmonic9_max = 0, 0.3
    harmonic11_min, harmonic11_max = 0, 0.3
    transient_amount_min, transient_amount_max = 0, 0
    transient_max_value_min, transient_max_value_max = 0, 0

    folder_name = 'generated_signals'
    os.makedirs(folder_name, exist_ok=True)

    mse_threshold = 0.01  # Define an acceptable MSE threshold for compression

    for i in range(num_files):
        # Linearly interpolate values between min and max for each parameter
        harmonic3 = harmonic3_min + (harmonic3_max - harmonic3_min) * (i + 1) / num_files
        harmonic5 = harmonic5_min + (harmonic5_max - harmonic5_min) * (i + 1) / num_files
        harmonic7 = harmonic7_min + (harmonic7_max - harmonic7_min) * (i + 1) / num_files
        harmonic9 = harmonic9_min + (harmonic9_max - harmonic9_min) * (i + 1) / num_files
        harmonic11 = harmonic11_min + (harmonic11_max - harmonic11_min) * (i + 1) / num_files
        transient_amount = int(
            transient_amount_min + (transient_amount_max - transient_amount_min) * (i + 1) / num_files)
        transient_max_value = transient_max_value_min + (transient_max_value_max - transient_max_value_min) * (
                    i + 1) / num_files

        # Generate the signal
        t, signal = generate_signal(num_samples, base_freq, sample_rate, amplitude, harmonic3, harmonic5, harmonic7,
                                    harmonic9, harmonic11, transient_amount, transient_max_value)

        # Optimize parameters for compression
        best_params = optimize_parameters(signal, mse_threshold)

        # Save the signal to a CSV file with optimal parameters in the filename
        wavelet = best_params['wavelet']
        level = best_params['level']
        threshold = best_params['threshold']
        filename = os.path.join(folder_name, f'signal{i}_{wavelet}_{level}_{threshold:.2f}.csv')
        save_to_csv(filename, t, signal)

    # Plot the first, middle, and last signal files
    plot_signals('generated_signals/signal*.csv')


if __name__ == "__main__":
    main()
