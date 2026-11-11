from filters import apply_blur_filters, apply_sharpening_kernel, test_on_noisy_image
from image_loader import (
    inspect_pixel_matrix,
    create_black_image,
    create_gradient_image,
    create_white_image,
    load_and_inspect_image,
    save_image_to_output
)
from transformations import perform_matrix_operations, calculate_image_statistics, explore_shapes_and_layouts, \
    resize_and_interpolate, flip_and_rotate_image, translate_image, convert_color_spaces, split_and_merge_channels, \
    adjust_brightness_contrast, generate_grayscale_histogram, apply_thresholding_methods, \
    apply_morphological_operations, detect_edges_canny, find_and_analyze_contours, run_batch_processing


def main():
    print("=== Running Complete Pipeline (Days 1 to 3) ===\n")

    # Day 1 Tasks
    inspect_pixel_matrix()
    print("-" * 50)
    create_black_image()
    print("-" * 50)
    create_gradient_image()
    print("=" * 50)
    create_white_image()
    print("=" * 50)

    # Day 2 Tasks
    perform_matrix_operations()
    print("-" * 50)
    calculate_image_statistics()
    print("-" * 50)
    explore_shapes_and_layouts()

    print("=" * 50)

    # Day 3 Tasks (OpenCV I/O)
    sample_input = "input_images/sample.jpg"
    sample_output = "output/grayscale/saved_sample.jpg"

    bgr_img, rgb_img = load_and_inspect_image(sample_input)
    if bgr_img is not None:
        save_image_to_output(bgr_img, sample_output)

    print("\nDay 3 OpenCV tasks completed successfully!")
    sample_input_a="input_images/astro.jpg"
    bgr_img1, rgb_img1 = load_and_inspect_image(sample_input_a)
    if bgr_img1 is not None:
        # 2. Run Day 4 Geometric Manipulations
        resize_and_interpolate(bgr_img1)
        flip_and_rotate_image(bgr_img1)
        translate_image(bgr_img1)
        # 2. Run Day 5 Colour & Channel Operations
        gray_img, hsv_img = convert_color_spaces(bgr_img1)
        split_and_merge_channels(bgr_img1)
        adjust_brightness_contrast(bgr_img1, alpha=1.2, beta=35)
        generate_grayscale_histogram(gray_img)
        apply_blur_filters(bgr_img1)
        apply_sharpening_kernel(bgr_img1)
        test_on_noisy_image()
    #     Day 7
        binary_mask = apply_thresholding_methods(bgr_img1)
        apply_morphological_operations(binary_mask)
        # 2. Run Day 8
        canny_mask = detect_edges_canny(bgr_img1)
        find_and_analyze_contours(bgr_img1, canny_mask)
        #Day 9
        run_batch_processing(input_dir="input_images/", base_output_dir="output/")

    else:
        print("Skipping geometric transforms because image loading failed.")



if __name__ == "__main__":
    main()