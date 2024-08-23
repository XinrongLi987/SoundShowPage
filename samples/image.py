#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :image.py
# @Time        :2024/8/23 上午3:30
# @Author      :InubashiriLix

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def calculate_spectrogram_range(wav_file):
    # Load the WAV file
    y, sr = librosa.load(wav_file, sr=None)
    # Generate the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    return D.min(), D.max()


def plot_spectrogram(wav_file, output_image, vmin, vmax):
    # Load the WAV file
    y, sr = librosa.load(wav_file, sr=None)
    # Generate the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Create the plot with fixed figure size
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis=None, y_axis=None, cmap='magma', vmin=vmin, vmax=vmax)
    plt.axis('off')  # Remove axes
    plt.gca().set_position([0, 0, 1, 1])  # Remove padding and make image fit the figure exactly
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


# Example usage
wav_files = ['1/leafStep.wav', '1/remix.wav', '1/output.wav']
output_images = ['1/sources.png', '1/remix.png', '1/output.png']

# Calculate min and max dB values across all files
dB_ranges = [calculate_spectrogram_range(wav_file) for wav_file in wav_files]
global_min_dB = min(dB_range[0] for dB_range in dB_ranges)
global_max_dB = max(dB_range[1] for dB_range in dB_ranges)

# Plot spectrograms with fixed scale
for wav_file, output_image in zip(wav_files, output_images):
    plot_spectrogram(wav_file, output_image, vmin=global_min_dB, vmax=global_max_dB)
