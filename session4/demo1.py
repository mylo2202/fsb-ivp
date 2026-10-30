import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the image and convert to grayscale
img_gray = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print("Error: Could not load '1.jpg'. Please check the file path.")
else:
    # 2. Convert to float (range 0.0 to 1.0)
    img_float = img_gray.astype(np.float32) / 255.0

    # 3. Define Edge Detection Kernels
    K_vert = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32)

    K_horiz = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    K_diag1 = np.array([
        [ 0,  1,  2],
        [-1,  0,  1],
        [-2, -1,  0]
    ], dtype=np.float32)

    K_diag2 = np.array([
        [ 2,  1,  0],
        [ 1,  0, -1],  # (Fixed minor sign consistency in template row)
        [ 0, -1, -2]
    ], dtype=np.float32)

    # 4. Apply filters using cv2.filter2D (-1 maintains the same depth/float type as input)
    res_vert  = cv2.filter2D(img_float, -1, K_vert)
    res_horiz = cv2.filter2D(img_float, -1, K_horiz)
    res_diag1 = cv2.filter2D(img_float, -1, K_diag1)
    res_diag2 = cv2.filter2D(img_float, -1, K_diag2)

    # 5. Visualize Results in a 2x2 grid
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(res_vert, cmap='gray')
    plt.title('Vertical Edges')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(res_horiz, cmap='gray')
    plt.title('Horizontal Edges')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(res_diag1, cmap='gray')
    plt.title('Diagonal Edges (+45°)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(res_diag2, cmap='gray')
    plt.title('Diagonal Edges (-45°)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()