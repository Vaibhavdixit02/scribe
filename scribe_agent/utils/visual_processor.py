import cv2
import numpy as np
import torch
import math
from PIL import Image
from transformers import AutoFeatureExtractor

class VisualProcessor:
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(vision_model_name)
        
    def preprocess_image(self, image_path, resize=(224, 224)):
        """Load and preprocess an image."""
        # Load image with cv2
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume image is already a numpy array
            image = image_path
            
        # Convert to PIL for compatibility with transformers
        pil_image = Image.fromarray(image)
        
        # Extract features using the model's preprocessor
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        
        return inputs
    
    def extract_visual_features(self, screenshot_data, element_bboxes):
        """Extract visual features and align with DOM elements.
        
        Args:
            screenshot_data: Either a path string or a dict with 'path' and 'bytes' keys
            element_bboxes: Dictionary mapping element IDs to bounding boxes
        """
        # Handle different screenshot_data formats
        if isinstance(screenshot_data, dict):
            # If we have the image bytes, decode them directly
            if 'bytes' in screenshot_data and screenshot_data['bytes']:
                image_bytes = screenshot_data['bytes']
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Otherwise try to load from path
            elif 'path' in screenshot_data:
                path = screenshot_data['path']
                image = cv2.imread(path)
            else:
                raise ValueError(f"Invalid screenshot_data dictionary: {screenshot_data}")
        elif isinstance(screenshot_data, str):
            # Traditional path loading
            image = cv2.imread(screenshot_data)
        else:
            raise ValueError(f"Unsupported screenshot_data type: {type(screenshot_data)}")  
        if image is None:
            raise ValueError(f"Failed to load image from {screenshot_data}")    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract visual features for each element
        element_features = {}
        full_image_inputs = self.preprocess_image(image)
        
        for element_id, bbox in element_bboxes.items():
            x, y, w, h = map(float, bbox.split(','))
            
            
            # Handle bbox values that might be out of image bounds
            x1 = int(max(0, math.floor(x)))
            y1 = int(max(0, math.floor(y)))
            x2 = int(min(math.floor(image.shape[1]), x + w))
            y2 = int(min(math.floor(image.shape[0]), y + h))
            
            # Skip if bbox is invalid
            if x1 >= x2 or y1 >= y2:
                continue

            # Extract region
            region = image[y1:y2, x1:x2]
            
            # Skip tiny regions
            if region.shape[0] < 4 or region.shape[1] < 4:
                continue
                
            # Preprocess region
            region_inputs = self.preprocess_image(region)
            
            # Store inputs for later feature extraction
            element_features[element_id] = region_inputs
        
        return {
            'full_image': full_image_inputs,
            'elements': element_features
        }
    
    def get_element_with_visual_context(self, html_str, node_id, visual_features):
        """Highlight the specified element in the HTML with visual context marker."""
        # This is a placeholder for actual integration logic
        # In practice, you would modify the HTML to add visual context markers
        # around the specified node or create a special token representation
        
        # For demonstration, we'll just add a marker string
        import re
        pattern = f'<[^>]* node="{node_id}"[^>]*>'
        marked_html = re.sub(
            pattern, 
            lambda m: m.group(0).replace('>', ' data-visual-context="true">'),
            html_str
        )
        
        return marked_html