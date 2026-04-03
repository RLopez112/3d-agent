# python src/yolov8_segmentation_all_elements.py
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# Install necessary libraries if you haven't already:
# pip install ultralytics opencv-python Pillow numpy matplotlib

def identify_and_crop_all_elements(image_path: str, output_dir: str = "segmented_objects"):
    """
    Identifies all elements in an image using YOLOv8-seg,
    and saves each detected object as a cropped image.
    """
    print(f"--- Identifying and Cropping All Elements for {image_path} ---")

    # 1. Load the image using OpenCV (YOLO prefers this format)
    try:
        # Read the image in BGR format (OpenCV default)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found or could not be read: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for PIL processing later
        print(f"Image loaded successfully: {image_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Create a dummy image if not found for demonstration
        print("Creating a dummy 'sample_image.jpg' for demonstration purposes.")
        dummy_img = Image.new('RGB', (1024, 768), color = 'lightblue')
        # Draw a few shapes
        for x in range(100, 300):
            for y in range(100, 200):
                dummy_img.putpixel((x, y), (255, 0, 0)) # Red square
        for x in range(400, 600):
            for y in range(300, 500):
                dummy_img.putpixel((x, y), (0, 255, 0)) # Green square
        for x in range(700, 900):
            for y in range(500, 700):
                dummy_img.putpixel((x, y), (0, 0, 255)) # Blue square
        
        dummy_img_path = "sample_image.jpg"
        dummy_img.save(dummy_img_path)
        print(f"Dummy image '{dummy_img_path}' created. Please re-run the script.")
        return

    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Load a pre-trained YOLOv8 segmentation model
    print("Loading YOLOv8s-seg model...")
    try:
        # 'yolov8s-seg.pt' is a smaller, faster segmentation model.
        # Use 'yolov8x-seg.pt' for higher accuracy but slower inference.
        model = YOLO("yolov8s-seg.pt")
        print("YOLOv8s-seg model loaded.")
    except Exception as e:
        print(f"Error loading YOLOv8s-seg model. Check internet connection or `ultralytics` installation: {e}")
        print("You might need to download the model file manually if behind a firewall.")
        return

    # 3. Run inference on the image
    print("Running YOLOv8-seg inference...")
    # 'results' will contain detections, including segmentation masks.
    # verbose=False suppresses detailed output from YOLO itself during inference
    results = model(img_bgr, verbose=False)
    print("Inference complete.")

    # 4. Process and save each detected object
    os.makedirs(output_dir, exist_ok=True)
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    objects_found = 0
    
    # Iterate through detections
    for r_idx, r in enumerate(results):
        if r.masks is None:
            print(f"No masks found for detection result {r_idx}. Skipping.")
            continue
            
        # r.boxes gives bounding boxes
        # r.masks gives segmentation masks
        # r.names maps class IDs to names (e.g., 0: 'person', 1: 'bicycle')

        for i, mask_tensor in enumerate(r.masks.xy):
            # Each `mask_tensor` is an array of [x, y] coordinates forming a polygon.
            # Convert to a binary mask image.
            
            # Get the bounding box for cropping (x_min, y_min, x_max, y_max)
            box = r.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Create a blank mask for the current object
            h, w, _ = img_bgr.shape
            binary_mask = np.zeros((h, w), dtype=np.uint8)

            # Fill the polygon defined by mask_tensor
            # YOLOv8 masks are typically scaled to original image size already
            segmentation_polygon = np.array([mask_tensor], dtype=np.int32)
            cv2.fillPoly(binary_mask, segmentation_polygon, 255) # Fill with white (255)

            # Apply the mask to the original image
            # Create a 4-channel image for transparency (RGBA)
            masked_object_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            masked_object_rgba[..., :3] = img_rgb # Copy RGB channels
            masked_object_rgba[..., 3] = binary_mask # Set alpha channel from binary mask

            # Crop the masked object using its bounding box
            cropped_object_rgba = masked_object_rgba[y1:y2, x1:x2, :]

            # Get class name and confidence
            class_id = int(r.boxes.cls[i])
            class_name = r.names[class_id]
            confidence = r.boxes.conf[i].item() # Convert tensor to Python float

            # Save the cropped image
            output_filename = f"{image_basename}_obj{objects_found:03d}_{class_name}_conf{confidence:.2f}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            
            # Use PIL to save with transparency
            Image.fromarray(cropped_object_rgba).save(output_filepath)
            
            print(f"Saved: {output_filepath} (Class: {class_name}, Confidence: {confidence:.2f})")
            objects_found += 1

    if objects_found == 0:
        print("No objects were detected in the image.")
    else:
        print(f"Successfully identified and saved {objects_found} objects.")
    
    # Optional: Display the original image with detections
    # Plotting is handled by YOLO's own `plot` method if `verbose=True`
    # or you can manually draw using OpenCV
    # r.plot() returns an image with detections drawn
    # Uncomment the following if you want a pop-up window:
    # annotated_frame = results[0].plot()
    # cv2.imshow("YOLOv8 Segmentation Results", annotated_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ensure you have an 'input_images' directory with an image named 'sample_image.jpg'
    # Or replace with your actual image path
    
    # This part will create a dummy image if `sample_image.jpg` doesn't exist
    # and then re-prompt to run the script.
    # The actual image loading is handled inside the function with a fallback.
    
    # Example Usage:
    # Replace 'sample_image.jpg' with the path to your actual image
    identify_and_crop_all_elements("sample_image.jpg")
# python src/yolov8_segmentation_all_elements.py
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# Install necessary libraries if you haven't already:
# pip install ultralytics opencv-python Pillow numpy matplotlib

def identify_and_crop_all_elements(image_path: str, output_dir: str = "segmented_objects"):
    """
    Identifies all elements in an image using YOLOv8-seg,
    and saves each detected object as a cropped image.
    """
    print(f"--- Identifying and Cropping All Elements for {image_path} ---")

    # 1. Load the image using OpenCV (YOLO prefers this format)
    try:
        # Read the image in BGR format (OpenCV default)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found or could not be read: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for PIL processing later
        print(f"Image loaded successfully: {image_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Create a dummy image if not found for demonstration
        print("Creating a dummy 'sample_image.jpg' for demonstration purposes.")
        dummy_img = Image.new('RGB', (1024, 768), color = 'lightblue')
        # Draw a few shapes
        for x in range(100, 300):
            for y in range(100, 200):
                dummy_img.putpixel((x, y), (255, 0, 0)) # Red square
        for x in range(400, 600):
            for y in range(300, 500):
                dummy_img.putpixel((x, y), (0, 255, 0)) # Green square
        for x in range(700, 900):
            for y in range(500, 700):
                dummy_img.putpixel((x, y), (0, 0, 255)) # Blue square
        
        dummy_img_path = "sample_image.jpg"
        dummy_img.save(dummy_img_path)
        print(f"Dummy image '{dummy_img_path}' created. Please re-run the script.")
        return

    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Load a pre-trained YOLOv8 segmentation model
    print("Loading YOLOv8s-seg model...")
    try:
        # 'yolov8s-seg.pt' is a smaller, faster segmentation model.
        # Use 'yolov8x-seg.pt' for higher accuracy but slower inference.
        model = YOLO("yolov8s-seg.pt")
        print("YOLOv8s-seg model loaded.")
    except Exception as e:
        print(f"Error loading YOLOv8s-seg model. Check internet connection or `ultralytics` installation: {e}")
        print("You might need to download the model file manually if behind a firewall.")
        return

    # 3. Run inference on the image
    print("Running YOLOv8-seg inference...")
    # 'results' will contain detections, including segmentation masks.
    # verbose=False suppresses detailed output from YOLO itself during inference
    results = model(img_bgr, verbose=False)
    print("Inference complete.")

    # 4. Process and save each detected object
    os.makedirs(output_dir, exist_ok=True)
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    objects_found = 0
    
    # Iterate through detections
    for r_idx, r in enumerate(results):
        if r.masks is None:
            print(f"No masks found for detection result {r_idx}. Skipping.")
            continue
            
        # r.boxes gives bounding boxes
        # r.masks gives segmentation masks
        # r.names maps class IDs to names (e.g., 0: 'person', 1: 'bicycle')

        for i, mask_tensor in enumerate(r.masks.xy):
            # Each `mask_tensor` is an array of [x, y] coordinates forming a polygon.
            # Convert to a binary mask image.
            
            # Get the bounding box for cropping (x_min, y_min, x_max, y_max)
            box = r.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Create a blank mask for the current object
            h, w, _ = img_bgr.shape
            binary_mask = np.zeros((h, w), dtype=np.uint8)

            # Fill the polygon defined by mask_tensor
            # YOLOv8 masks are typically scaled to original image size already
            segmentation_polygon = np.array([mask_tensor], dtype=np.int32)
            cv2.fillPoly(binary_mask, segmentation_polygon, 255) # Fill with white (255)

            # Apply the mask to the original image
            # Create a 4-channel image for transparency (RGBA)
            masked_object_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            masked_object_rgba[..., :3] = img_rgb # Copy RGB channels
            masked_object_rgba[..., 3] = binary_mask # Set alpha channel from binary mask

            # Crop the masked object using its bounding box
            cropped_object_rgba = masked_object_rgba[y1:y2, x1:x2, :]

            # Get class name and confidence
            class_id = int(r.boxes.cls[i])
            class_name = r.names[class_id]
            confidence = r.boxes.conf[i].item() # Convert tensor to Python float

            # Save the cropped image
            output_filename = f"{image_basename}_obj{objects_found:03d}_{class_name}_conf{confidence:.2f}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            
            # Use PIL to save with transparency
            Image.fromarray(cropped_object_rgba).save(output_filepath)
            
            print(f"Saved: {output_filepath} (Class: {class_name}, Confidence: {confidence:.2f})")
            objects_found += 1

    if objects_found == 0:
        print("No objects were detected in the image.")
    else:
        print(f"Successfully identified and saved {objects_found} objects.")
    
    # Optional: Display the original image with detections
    # Plotting is handled by YOLO's own `plot` method if `verbose=True`
    # or you can manually draw using OpenCV
    # r.plot() returns an image with detections drawn
    # Uncomment the following if you want a pop-up window:
    # annotated_frame = results[0].plot()
    # cv2.imshow("YOLOv8 Segmentation Results", annotated_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ensure you have an 'input_images' directory with an image named 'sample_image.jpg'
    # Or replace with your actual image path
    
    # This part will create a dummy image if `sample_image.jpg` doesn't exist
    # and then re-prompt to run the script.
    # The actual image loading is handled inside the function with a fallback.
    
    # Example Usage:
    # Replace 'sample_image.jpg' with the path to your actual image
    identify_and_crop_all_elements("../tmp/images.jpeg")
