import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image and compute 2D FFT
I = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

if I is None:
    print("Error: Could not load 'cameraman.tif'. Please check the file path.")
else:
    Id = I.astype(np.float64) / 255.0
    ft = np.fft.fft2(Id)
    ft_shift = np.fft.fftshift(ft)

    # Magnitude spectra
    mag = np.abs(ft_shift)
    log_mag = np.log(1.0 + mag)

    # Display comparisons
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mag, cmap='gray')
    plt.title('Direct Remap')

    plt.subplot(1, 2, 2)
    plt.imshow(log_mag, cmap='gray')
    plt.title('Log Remap')
    plt.tight_layout()
    plt.show()

    # 2. Distance matrix creation
    M, N = I.shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D_shift = np.sqrt(U**2 + V**2)