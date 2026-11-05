import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
def perform_matrix_operations():
    print("=== Exercise 1: Matrix Arithmetic & ROI Extraction ===")

    # 1. Create a synthetic 5x5 grayscale image matrix
    img = np.array([
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],
        [110, 120, 130, 140, 150],
        [160, 170, 180, 190, 200],
        [210, 220, 230, 240, 250]
    ], dtype=np.uint8)

    print(f"Original Shape: {img.shape}, Dtype: {img.dtype}")

    # 2. Element-wise Arithmetic (Adjusting Brightness via Broadcasting)
    # Using np.clip to prevent uint8 overflow (keeping values between 0 and 255)
    brighter_img = np.clip(img.astype(np.int16) + 30, 0, 255).astype(np.uint8)
    print("\nBrighter Image (Added 30 to all pixels):")
    print(brighter_img)

    # 3. Extracting a Rectangular Region (ROI / Cropping)
    # Let's crop the center 3x3 region (rows 1 to 3, columns 1 to 3)
    roi = img[1:4, 1:4]
    print("\nExtracted Region of Interest (ROI - Center 3x3):")
    print(roi)

    # 4. Modify selected pixels in the ROI
    roi[1, 1] = 255  # Set the absolute center pixel to pure white
    print("\nModified Original Image after changing ROI center:")
    print(img)
    cv2.imwrite(os.path.join("output/filtered/", "roi.jpg"), img)
    print("*******************************")


def calculate_image_statistics():
    print("\n=== Exercise 2: Image Statistics (Min, Max, Mean, Std) ===")

    # Create a random image matrix
    np.random.seed(42)  # For reproducible random numbers
    random_img = np.random.randint(0, 255, size=(4, 4), dtype=np.uint8)
    print("Random Image Matrix:")
    print(random_img)

    print(f"Minimum Intensity: {random_img.min()}")
    print(f"Maximum Intensity: {random_img.max()}")
    print(f"Mean Brightness: {random_img.mean():.2f}")
    print(f"Standard Deviation (Contrast): {random_img.std():.2f}")

    # Axis operations: Mean across rows (axis=1) vs columns (axis=0)
    print(f"Mean across columns (axis=0): {random_img.mean(axis=0)}")
    print(f"Mean across rows (axis=1): {random_img.mean(axis=1)}")
    print("************************************")

def explore_shapes_and_layouts():
    print("\n=== Exercise 3: Transpose, Reshape, and Flatten ===")

    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.uint8)

    print("Original Matrix (2x3):\n", matrix)

    # Transpose (.T)
    print("Transposed Matrix (3x2):\n", matrix.T)

    # Flatten (Converting multi-dimensional array into 1D)
    flattened = matrix.flatten()
    print("Flattened 1D Array:", flattened)

    # Reshape (Back to 3x2)
    reshaped = flattened.reshape((3, 2))
    print("Reshaped back to (3x2):\n", reshaped)


# ==========================================
# DAY 4: BASIC IMAGE MANIPULATION FUNCTIONS
# ==========================================

def resize_and_interpolate(image_bgr, output_dir="output/resized/"):
    print("\n=== Day 4: Resizing and Interpolation Methods ===")
    os.makedirs(output_dir, exist_ok=True)

    h, w = image_bgr.shape[:2]
    print(f"Original Dimensions: Width={w}, Height={h}")

    # 1. Resize smaller using Bilinear Interpolation (Smooth standard default)
    resized_linear = cv2.resize(image_bgr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_dir, "resized_linear.jpg"), resized_linear)

    # 2. Resize smaller using Nearest Neighbor (Blocky, pixel art style)
    resized_nearest = cv2.resize(image_bgr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(output_dir, "resized_nearest.jpg"), resized_nearest)

    # 3. Resize larger (Zoom in 2x) using Bicubic Interpolation (High quality edges)
    resized_cubic = cv2.resize(image_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, "resized_cubic_zoom.jpg"), resized_cubic)

    print("Successfully saved interpolation test images to output/resized/")


def flip_and_rotate_image(image_bgr, output_dir="output/resized/"):
    print("\n=== Day 4: Flipping and Rotating Images ===")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Horizontal Flip (Mirror image: flipCode = 1)
    flipped_h = cv2.flip(image_bgr, 1)
    cv2.imwrite(os.path.join(output_dir, "flipped_horizontal.jpg"), flipped_h)

    # 2. Vertical Flip (Upside down: flipCode = 0)
    flipped_v = cv2.flip(image_bgr, 0)
    cv2.imwrite(os.path.join(output_dir, "flipped_vertical.jpg"), flipped_v)

    # 3. Rotation (Rotate 45 degrees clockwise around the center)
    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    rotated = cv2.warpAffine(image_bgr, rotation_matrix, (w, h))
    cv2.imwrite(os.path.join(output_dir, "rotated_45.jpg"), rotated)

    print("Successfully saved flipped and rotated images to output/resized/")


def translate_image(image_bgr, output_dir="output/resized/"):
    print("\n=== Day 4: Image Translation (Shifting) ===")
    os.makedirs(output_dir, exist_ok=True)

    h, w = image_bgr.shape[:2]
    # Translation Matrix M: shift x by +40 pixels, shift y by +60 pixels
    M = np.float32([[1, 0, 40], [0, 1, 60]])
    translated = cv2.warpAffine(image_bgr, M, (w, h))
    cv2.imwrite(os.path.join(output_dir, "translated.jpg"), translated)

    print("Successfully saved translated image to output/resized/")

#Day 5
def convert_color_spaces(image_bgr, output_dir="output/filtered/"):
    print("\n=== Day 5: Colour Space Conversions (BGR, RGB, Grayscale, HSV) ===")
    os.makedirs(output_dir, exist_ok=True)

    # 1. BGR to Grayscale
    gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "gray_image.jpg"), gray_img)

    # 2. BGR to RGB (Standard color space)
    rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Note: OpenCV saves images in BGR, so if we save rgb_img directly using cv2.imwrite,
    # red and blue will swap. To save RGB correctly via OpenCV, convert back or use cv2.cvtColor
    cv2.imwrite(os.path.join(output_dir, "rgb_converted.jpg"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    # 3. BGR to HSV (Hue, Saturation, Value)
    hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(output_dir, "hsv_image.jpg"), hsv_img)

    print("Successfully saved color-converted images to output/filtered/")
    return gray_img, hsv_img


def split_and_merge_channels(image_bgr, output_dir="output/filtered/"):
    print("\n=== Day 5: Splitting and Merging Channels ===")
    os.makedirs(output_dir, exist_ok=True)

    # Split the 3D BGR image into 3 separate 2D channels
    b, g, r = cv2.split(image_bgr)

    # Save individual channels as grayscale visual representations
    cv2.imwrite(os.path.join(output_dir, "channel_blue.jpg"), b)
    cv2.imwrite(os.path.join(output_dir, "channel_green.jpg"), g)
    cv2.imwrite(os.path.join(output_dir, "channel_red.jpg"), r)

    # Merge them back together
    merged_bgr = cv2.merge([b, g, r])
    print(f"Channels successfully split and merged back. Merged shape: {merged_bgr.shape}")
    return merged_bgr


def adjust_brightness_contrast(image_bgr, alpha=1.3, beta=40, output_dir="output/filtered/"):
    print(f"\n=== Day 5: Brightness (beta={beta}) and Contrast (alpha={alpha}) ===")
    os.makedirs(output_dir, exist_ok=True)

    # Formula: g(x) = alpha * f(x) + beta
    # Using np.clip to prevent overflow beyond 255 or underflow below 0
    adjusted = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)

    cv2.imwrite(os.path.join(output_dir, "brightness_contrast_adjusted.jpg"), adjusted)
    print("Successfully saved brightness/contrast adjusted image to output/filtered/")
    return adjusted


def generate_grayscale_histogram(gray_image, output_dir="output/filtered/"):
    print("\n=== Day 5: Generating Grayscale Histogram ===")
    os.makedirs(output_dir, exist_ok=True)

    # Calculate histogram using OpenCV
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])#doubt

    # Plot and save histogram using Matplotlib
    plt.figure(figsize=(8, 4))
    plt.title("Grayscale Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Pixel Count")
    plt.plot(hist, color='black')
    plt.xlim([0, 256])

    hist_path = os.path.join(output_dir, "grayscale_histogram.png")
    plt.savefig(hist_path)
    plt.close()

    print(f"Histogram successfully plotted and saved to: {hist_path}")


if __name__ == "__main__":
    perform_matrix_operations()
    calculate_image_statistics()
    explore_shapes_and_layouts()
    sample_path = "input_images/astro.jpg"
    bgr_img=cv2.imread(sample_path, cv2.IMREAD_COLOR)
    if bgr_img is not None:
        # 2. Run Day 5 Colour & Channel Operations
        gray_img, hsv_img = convert_color_spaces(bgr_img)
        split_and_merge_channels(bgr_img)
        adjust_brightness_contrast(bgr_img, alpha=1.2, beta=35)
        generate_grayscale_histogram(gray_img)
    else:
        print("Skipping color tasks because image loading failed.")

    print("\nDay 5 colour space and channel tasks completed successfully!")
