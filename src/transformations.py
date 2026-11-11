import csv
import glob

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
#Day 7
def apply_thresholding_methods(image_bgr, output_dir="output/masks/"):
    print("\n=== Day 7: Binary, Adaptive, and Otsu Thresholding ===")
    os.makedirs(output_dir, exist_ok=True)

    # Convert to grayscale first (thresholding operates on single-channel images)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Global Binary Thresholding (Cutoff at 127)
    _, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, "thresh_binary.jpg"), thresh_binary)

    # 2. Global Inverse Binary Thresholding
    _, thresh_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(output_dir, "thresh_inv.jpg"), thresh_inv)

    # 3. Adaptive Gaussian Thresholding (Handles uneven lighting)
    # Parameters: src, maxValue, adaptiveMethod, thresholdType, blockSize, C constant
    thresh_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite(os.path.join(output_dir, "thresh_adaptive.jpg"), thresh_adaptive)

    # 4. Otsu's Automated Thresholding (Calculates optimal threshold automatically)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, "thresh_otsu.jpg"), thresh_otsu)

    print("Successfully saved threshold masks to output/masks/")
    return thresh_otsu

def apply_morphological_operations(binary_mask, output_dir="output/masks/"):
    print("\n=== Day 7: Morphological Operations (Erosion, Dilation, Opening, Closing) ===")
    os.makedirs(output_dir, exist_ok=True)

    # Define a structuring element (kernel), e.g., a 3x3 rectangular matrix
    kernel = np.ones((3, 3), dtype=np.uint8)

    # 1. Erosion (Shrinks white regions, removes tiny speckle noise)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "morph_eroded.jpg"), eroded)

    # 2. Dilation (Expands white regions, fills small internal holes)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "morph_dilated.jpg"), dilated)

    # 3. Opening (Erosion followed by Dilation -> removes external noise)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_dir, "morph_opening.jpg"), opening)

    # 4. Closing (Dilation followed by Erosion -> closes internal gaps)
    closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, "morph_closing.jpg"), closing)

    print("Successfully saved morphological operation results to output/masks/")
#Day 8
def detect_edges_canny(image_bgr, output_dir="output/edges/"):
    print("\n=== Day 8: Gradient Operators & Canny Edge Detection ===")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian Blur first to reduce noise (prevents false edges)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # 3. Sobel Operators (Horizontal and Vertical Gradients)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # X direction
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Y direction

    # Convert back to uint8 absolute values
    sobel_x_uint8 = cv2.convertScaleAbs(sobel_x)
    sobel_y_uint8 = cv2.convertScaleAbs(sobel_y)

    cv2.imwrite(os.path.join(output_dir, "edge_sobel_x.jpg"), sobel_x_uint8)
    cv2.imwrite(os.path.join(output_dir, "edge_sobel_y.jpg"), sobel_y_uint8)

    # 4. Canny Edge Detection (Golden Standard)
    # Parameters: image, threshold1 (minVal), threshold2 (maxVal)
    canny_edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    cv2.imwrite(os.path.join(output_dir, "edge_canny.jpg"), canny_edges)

    print("Successfully saved edge detection maps to output/edges/")
    return canny_edges


def find_and_analyze_contours(image_bgr, edge_mask, output_dir="output/edges/"):
    print("\n=== Day 8: Finding Contours, Areas, Perimeters & Bounding Boxes ===")
    os.makedirs(output_dir, exist_ok=True)

    # Make a copy of the original image to draw bounding boxes onto
    contour_display_img = image_bgr.copy()

    # Find contours from the edge mask
    # cv2.RETR_EXTERNAL retrieves only outer contours; CHAIN_APPROX_SIMPLE compresses horizontal/vertical segments
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total raw contours found: {len(contours)}")

    valid_contours_count = 0

    for idx, cnt in enumerate(contours):
        # Calculate area and perimeter
        area = cv2.contourArea(cnt)

        # Filter out tiny noise contours by setting a minimum area threshold (e.g., > 500 pixels)
        if area > 500:
            valid_contours_count += 1
            perimeter = cv2.arcLength(cnt, closed=True)

            # Get bounding rectangle coordinates (x, y, width, height)
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw the contour outline in green
            cv2.drawContours(contour_display_img, [cnt], -1, (0, 255, 0), 2)

            # Draw the bounding rectangle in blue
            cv2.rectangle(contour_display_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Put text label for area inspection
            label = f"ID:{valid_contours_count} A:{int(area)}"
            cv2.putText(contour_display_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save the final annotated contour image
    output_path = os.path.join(output_dir, "annotated_contours.jpg")
    cv2.imwrite(output_path, contour_display_img)

    print(f"Filtered down to {valid_contours_count} significant object contours.")
    print(f"Successfully saved annotated contours image to: {output_path}")


def segment_by_color_hsv(image_bgr, lower_bound, upper_bound):
    """
    Creates a binary mask isolating a specific color range in HSV space.
    """
    # 1. Convert BGR to HSV color space
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 2. Create a mask using the specified color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 3. Clean up noise using morphological opening/closing
    kernel = np.ones((5, 5), dtype=np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Bitwise-AND to extract the segmented region from the original image
    segmented_result = cv2.bitwise_and(image_bgr, image_bgr, mask=cleaned_mask)

    return cleaned_mask, segmented_result

##Day 9
def run_batch_processing(input_dir="input_images/", base_output_dir="output/"):
    print(f"\n=== Day 9: Running Automated Batch Processing on folder: '{input_dir}' ===")

    # Define sub-output folders based on your project structure
    mask_out_dir = os.path.join(base_output_dir, "masks")
    filtered_out_dir = os.path.join(base_output_dir, "filtered")
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(filtered_out_dir, exist_ok=True)

    # Find all common image formats (.jpg, .jpeg, .png) in the input directory
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"Warning: No images found in '{input_dir}' directory!")
        return

    print(f"Found {len(image_paths)} images to process.")

    # Example HSV bounds for filtering (e.g., targeting a specific color range like bluish/greenish tones)
    # You can tweak these lower and upper bounds depending on your test images
    lower_hsv = np.array([90, 10, 100], dtype=np.uint8)
    upper_hsv = np.array([130, 255, 255], dtype=np.uint8)

    processed_count = 0
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing file: {filename}...")

        # Load image safely
        img = cv2.imread(img_path)
        if img is None:
            print(f"  -> Failed to load {filename}, skipping.")
            continue

        # Apply HSV color segmentation
        mask, segmented = segment_by_color_hsv(img, lower_hsv, upper_hsv)

        # Save mask and processed results into their designated directories
        cv2.imwrite(os.path.join(mask_out_dir, f"mask_{filename}"), mask)
        cv2.imwrite(os.path.join(filtered_out_dir, f"segmented_{filename}"), segmented)

        processed_count += 1

    print(f"Batch processing complete! Successfully processed {processed_count} images.")

#Day 10
def run_complete_mini_project_pipeline(input_dir="input_images/", base_output_dir="output/",
                                       report_path="reports/processing_report.csv"):
    print("\n=== Day 10: Running Complete End-to-End Preprocessing Mini-Project ===")

    # 1. Ensure all manager-specified output folders and report directories exist
    os.makedirs(os.path.join(base_output_dir, "resized"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "grayscale"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "filtered"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "edges"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # 2. Gather all target images using glob
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"Error: No images found in '{input_dir}' directory to process.")
        return

    print(f"Found {len(image_paths)} images. Starting automated batch processing & telemetry logging...")

    # 3. Open CSV file to write the processing telemetry report
    report_file = open(report_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(report_file)

    # Write CSV Header matching enterprise reporting standards
    csv_writer.writerow(
        ["Filename", "Original_Width", "Original_Height", "Mean_Brightness", "Contours_Detected", "Status"])

    success_count = 0

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"\n[Processing] -> {filename}")

        # Input Validation / Safety Check
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [Error] Failed to read {filename}. Skipping file.")
            csv_writer.writerow([filename, 0, 0, 0.0, 0, "FAILED_READ"])
            continue

        h, w = img.shape[:2]

        try:
            # Step A: Grayscale Conversion & Saving
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(base_output_dir, "grayscale", f"gray_{filename}"), gray)

            # Step B: Calculate Mean Brightness Telemetry
            mean_brightness = float(np.mean(gray))

            # Step C: Resize (e.g., scale down by 50% while preserving aspect ratio)
            resized = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(base_output_dir, "resized", f"resized_{filename}"), resized)

            # Step D: Filtering & Denoising (Gaussian Blur)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            cv2.imwrite(os.path.join(base_output_dir, "filtered", f"blur_{filename}"), blurred)

            # Step E: Edge Detection (Canny) & Contour Analysis
            edges = cv2.Canny(blurred, 100, 200)
            cv2.imwrite(os.path.join(base_output_dir, "edges", f"canny_{filename}"), edges)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            significant_contours = sum(1 for c in contours if cv2.contourArea(c) > 100)

            # Step F: Thresholding & Binary Mask Generation
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(base_output_dir, "masks", f"mask_{filename}"), mask)

            # Log successful processing row into CSV
            csv_writer.writerow([filename, w, h, round(mean_brightness, 2), significant_contours, "SUCCESS"])
            success_count += 1
            print(f"  [Complete] W: {w}, H: {h} | Brightness: {mean_brightness:.1f} | Contours: {significant_contours}")

        except Exception as e:
            print(f"  [Error] Exception occurred during processing {filename}: {e}")
            csv_writer.writerow([filename, w, h, 0.0, 0, f"ERROR: {str(e)}"])

    # Close report file securely
    report_file.close()
    print(f"\n=== Mini-Project Pipeline Complete! ===")
    print(f"Successfully processed {success_count}/{len(image_paths)} images.")
    print(f"Telemetry report generated and saved to: {report_path}")

