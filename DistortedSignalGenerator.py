import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# Function to generate a signal with specified parameters
def generate_signal(num_samples, base_freq, sample_rate, amplitude, harmonic3, harmonic5, harmonic7, harmonic9,
                    harmonic11, transient_amount, transient_max_value):
    # Generate time vector
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)

    # Generate base signal (sine wave)
    base_signal = amplitude * np.sin(2 * np.pi * base_freq * t)

    # Generate harmonic components and add to base signal
    harmonic_signal = (harmonic3 * amplitude * np.sin(2 * np.pi * 3 * base_freq * t) +
                       harmonic5 * amplitude * np.sin(2 * np.pi * 5 * base_freq * t) +
                       harmonic7 * amplitude * np.sin(2 * np.pi * 7 * base_freq * t) +
                       harmonic9 * amplitude * np.sin(2 * np.pi * 9 * base_freq * t) +
                       harmonic11 * amplitude * np.sin(2 * np.pi * 11 * base_freq * t))

    signal = base_signal + harmonic_signal

    # Add transient noise to random samples
    transient_indices = np.random.choice(num_samples, transient_amount, replace=False)
    for idx in transient_indices:
        signal[idx] += transient_max_value * np.random.uniform(-1, 1)

    return t, signal


# Function to save signal data to a CSV file
def save_to_csv(filename, t, signal):
    df = pd.DataFrame({'Time': t, 'Signal': signal})
    df.to_csv(filename, index=False)


# Function to plot signals from CSV files
def plot_signals(filenames):
    plt.figure(figsize=(15, 5))
    for i, filename in enumerate(filenames):
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


# Main function to generate signals and save them to files
def main():
    num_files = 100  # Number of signal files to generate
    num_samples = 1000  # Number of samples per signal
    base_freq = 50  # Base frequency in Hz
    sample_rate = 5000  # Sample rate in Hz
    amplitude = 230  # Amplitude of the signal

    # Define min and max values for harmonics and transient parameters
    harmonic3_min, harmonic3_max = 0, 0.3
    harmonic5_min, harmonic5_max = 0, 0.35
    harmonic7_min, harmonic7_max = 0, 0.32
    harmonic9_min, harmonic9_max = 0, 0.31
    harmonic11_min, harmonic11_max = 0.25, 0.305
    transient_amount_min, transient_amount_max = 10, 50
    transient_max_value_min, transient_max_value_max = 0.2, 0.8

    folder_name = 'generated_signals'
    os.makedirs(folder_name, exist_ok=True)

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

        # Save the signal to a CSV file
        filename = os.path.join(folder_name, f'signal_{i + 1}.csv')
        save_to_csv(filename, t, signal)

    # Plot the first, middle, and last signal files
    plot_signals([
        os.path.join(folder_name, 'signal_1.csv'),
        os.path.join(folder_name, f'signal_{num_files // 2}.csv'),
        os.path.join(folder_name, f'signal_{num_files}.csv')
    ])


if __name__ == "__main__":
    main()
