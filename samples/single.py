#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :single_image.py
# @Time        :2024/8/23 上午3:30
# @Author      :InubashiriLix

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(wav_file, output_image):
    # Load the WAV file
    y, sr = librosa.load(wav_file, sr=None)
    # Generate the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Create the plot with fixed figure size
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis=None, y_axis=None, cmap='magma')
    plt.axis('off')  # Remove axes
    plt.gca().set_position([0, 0, 1, 1])  # Remove padding and make image fit the figure exactly
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


# Example usage
wav_file = '1/leafStep1.wav'  # Replace with your WAV file path
output_image = '1/test.png'  # Replace with your output image path

# Plot the spectrogram
plot_spectrogram(wav_file, output_image)
