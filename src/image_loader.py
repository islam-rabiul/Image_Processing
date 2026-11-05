import numpy as np
import cv2
import os

def inspect_pixel_matrix():
    print("=== Day 1: 5x5 Pixel Matrix Inspection ===")

    # 1. Create a custom 5x5 grayscale pixel matrix with integer values between 0 and 255
    # Using unsigned 8-bit integers (uint8) which is the standard format for images
    pixel_matrix=np.array([[24,45,6,55,45],[23,4,5,3,2],[2,3,4,5,6],[45,6,5,4,6],[234,45,6,7,7]],dtype=np.uint8)
    # 2. Inspect shape, dtype, min, and max values
    print(pixel_matrix)
    print(pixel_matrix.shape)
    print(pixel_matrix.dtype)
    print(pixel_matrix.min())
    print(pixel_matrix.max())

    # 3. Modify selected pixels (e.g., set the center pixel to 0 - black)
    print("\nOriginal Center Pixel Value:", pixel_matrix[2, 2])
    pixel_matrix[2, 2] = 0
    print("Modified Center Pixel Value (set to 0):", pixel_matrix[2, 2])

    print("\nUpdated Matrix:")
    print(pixel_matrix)
    return pixel_matrix

def create_black_image():
    pixel_matrix=np.zeros((100,100),dtype=np.uint8)
    print(f"Black Image Created ,Minimum pixel value: {pixel_matrix.min()} ,Maximum pixel value: {pixel_matrix.max()}")
def create_white_image():
    pixel_matrix=np.ones((100,100),dtype=np.uint8)
    print(f"White Image Created ,Minimum pixel value: {pixel_matrix.min()} ,Maximum pixel value: {pixel_matrix.max()}")
def create_gradient_image():
    row_gradient=np.linspace(0,255,num=100,dtype=np.uint8)
    gradient_image=np.tile(row_gradient,(100,1))
    print(f"Gradient Image -> Shape: {gradient_image.shape}, Min: {gradient_image.min()}, Max: {gradient_image.max()}")


# Day 3 and 4
def load_and_inspect_image(image_path):
    print("=== OpenCV Image Loading & Property Inspection ===")

    # Check if the file exists. If not, generate a dummy sample image for testing.
    if not os.path.exists(image_path):
        print(f"Sample image not found at '{image_path}'. Creating a temporary test image...")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # Create a 100x100 BGR image (Blue square)
        dummy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        #
        # dummy_img[:, :] = [154, 145, 100]  # BGR format: Blue = 255
        cv2.imwrite(image_path, dummy_img)

    # 1. Read the image in Color mode
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 2. Safety Check: Handle failed image reads
    if img_bgr is None:
        print(f"Error: Failed to load image from {image_path}. Check file path or format.")
        return None

    # 3. Inspect Image Properties
    print(f"File Path: {image_path}")
    print(f"Image Shape (Height, Width, Channels): {img_bgr.shape}")
    print(f"Data Type (dtype): {img_bgr.dtype}")
    print(f"Total Number of Pixels (Size): {img_bgr.size}")

    # Sample individual pixel value at row 50, col 50 (Returns [B, G, R])
    print(f"Pixel value at [50, 50] (BGR): {img_bgr[50, 50]}")

    # 4. BGR to RGB Conversion (Crucial for correct display in libraries like matplotlib/Jupyter)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Converted to RGB. Pixel value at [50, 50] (RGB): {img_rgb[50, 50]}")

    return img_bgr, img_rgb


def save_image_to_output(image_matrix, output_path):
    print("\n=== OpenCV Image Saving ===")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save using cv2.imwrite
    success = cv2.imwrite(output_path, image_matrix)

    if success:
        print(f"Success! Image safely stored at: {output_path}")
    else:
        print(f"Error: Failed to save image to {output_path}")


if __name__ == "__main__":
    sample_path = "input_images/sample1.jpg"
    out_path = "output/grayscale/sample_output1.jpg"

    loaded_bgr, loaded_rgb = load_and_inspect_image(sample_path)
    if loaded_bgr is not None:
        save_image_to_output(loaded_bgr, out_path)

# Day 3 and 4 End **********************#############################***************************

