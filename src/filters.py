import numpy as np
import cv2
import os


def apply_blur_filters(image_bgr, output_dir="output/filtered/"):
    print("\n=== Day 6: Applying Blur Filters and Comparing Kernels ===")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Averaging / Box Blur (Kernel size 5x5)
    avg_blur = cv2.blur(image_bgr, (5, 5))
    cv2.imwrite(os.path.join(output_dir, "blur_averaging.jpg"), avg_blur)

    # 2. Gaussian Blur (Kernel size 5x5, standard deviation 0)
    gaussian_blur = cv2.GaussianBlur(image_bgr, (5, 5), sigmaX=0)
    cv2.imwrite(os.path.join(output_dir, "blur_gaussian.jpg"), gaussian_blur)

    # 3. Median Blur (Great for salt-and-pepper noise, aperture size must be odd, e.g., 5)
    median_blur = cv2.medianBlur(image_bgr, 5)
    cv2.imwrite(os.path.join(output_dir, "blur_median.jpg"), median_blur)

    # 4. Bilateral Filter (Preserves edges while smoothing flat regions)
    # Parameters: source, d (pixel neighborhood diameter), sigmaColor, sigmaSpace
    bilateral = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite(os.path.join(output_dir, "blur_bilateral.jpg"), bilateral)

    print("Successfully saved various blur filter outputs to output/filtered/")


def apply_sharpening_kernel(image_bgr, output_dir="output/filtered/"):
    print("\n=== Day 6: Creating and Applying a Sharpening Kernel ===")
    os.makedirs(output_dir, exist_ok=True)

    # Define a standard 3x3 sharpening kernel matrix
    # Center pixel is high positive, surrounding neighbors are negative
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # Apply the kernel using cv2.filter2D
    sharpened = cv2.filter2D(image_bgr, ddepth=-1, kernel=sharpening_kernel)

    cv2.imwrite(os.path.join(output_dir, "sharpened_image.jpg"), sharpened)
    print("Successfully saved sharpened image to output/filtered/")
    return sharpened


def test_on_noisy_image(output_dir="output/filtered/"):
    print("\n=== Day 6: Testing Filters on Synthetic Noisy Image ===")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create a clean synthetic image (gray box with text)
    clean_img = np.full((200, 200, 3), 128, dtype=np.uint8)
    cv2.putText(clean_img, "AI LEARNERS", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 2. Add artificial random noise using pure NumPy (safely clipped to 0-255 uint8)
    noise = np.random.randint(-40, 40, clean_img.shape, dtype=np.int16)
    noisy_img = np.clip(clean_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "synthetic_noisy.jpg"), noisy_img)

    # 3. Apply Median Blur to remove the noise while preserving structure
    denoised_img = cv2.medianBlur(noisy_img, 5)
    cv2.imwrite(os.path.join(output_dir, "synthetic_denoised.jpg"), denoised_img)

    print("Successfully generated noisy image test results in output/filtered/")