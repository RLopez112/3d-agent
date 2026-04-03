# python src/image_loader.py
from PIL import Image
# python src/segmentation_module.py
from transformers import SamModel, SamProcessor
import torch
import numpy as np
from PIL import Image

def load_image(image_path: str):
    """Loads an image from the given path."""
    try:
        img = Image.open(image_path).convert("RGB") # Ensure RGB format
        print(f"Successfully loaded image from {image_path}")
        return img
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

#... existing code ...



def get_segmented_objects(image: Image.Image):
    """
    Performs segmentation on the input image using a pre-trained model
    and returns a list of segmented objects.
    """
    # Load SAM model and processor (only once, or outside this function for efficiency)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    # Convert image to tensor
    inputs = processor.preprocess(image)
    # inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Decode masks (this part can be complex depending on the model output)
    masks = processor.decode(outputs.pred_masks.squeeze(1)) # Example for SAM

    segmented_objects = []
    for i, mask_tensor in enumerate(masks):
        mask_np = mask_tensor.squeeze().cpu().numpy()
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))

        # Create a blank image to paste the segmented object
        object_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
        # Use the mask to paste the relevant part of the original image
        object_img.paste(image, (0, 0), mask_image)

        # Calculate bounding box
        bbox = mask_image.getbbox() # Returns (left, upper, right, lower)

        # Create a dictionary for the segmented object
        segmented_objects.append({
            "id": f"obj_{i:03d}",
            "name": f"Object {i+1}", # Simple naming, could be improved with detection
            "mask": mask_np,
            "bbox": bbox,
            "segment_image": object_img.crop(bbox) # Crop to tight bbox
        })
    print(f"Found {len(segmented_objects)} objects.")
    return segmented_objects

# ... existing code ...

get_segmented_objects(load_image("../tmp/b.png"))