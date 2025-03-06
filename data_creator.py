import pickle
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, HTML
from transformers import AutoTokenizer
# Import from our package
from scribe_agent.models.cross_modal_model import CrossModalWebAgent
from scribe_agent.utils.visual_processor import VisualProcessor
from scribe_agent.utils.html_processor import process_html

# Set the model path (update this to your trained model path)
MODEL_PATH = "checkpoints/best_model"  # Change to your actual model path

# For testing without a trained model, use the base models directly
TEXT_MODEL = "../Qwen2.5-7B-Instruct"
VISION_MODEL = "clip-vit-base-patch32"

# Check if we have a trained model
has_trained_model = os.path.exists(MODEL_PATH)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer
if has_trained_model:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
else:
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    
# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize model
if has_trained_model:
    model = CrossModalWebAgent.from_pretrained(MODEL_PATH).to(device)
else:
    # For testing only: creates a new model without fine-tuning
    model = CrossModalWebAgent(
        text_model_name=TEXT_MODEL,
        vision_model_name=VISION_MODEL,
        use_lora=False  # Don't use LoRA for testing
    )
    
model.eval()

# Initialize visual processor
visual_processor = VisualProcessor(vision_model_name=VISION_MODEL)

from scribe_agent.data.mind2web_dataset import create_multimodal_mind2web_dataloader

num_processors = os.cpu_count()

def custom_collate_fn(batch):
    """Custom collate function to handle complex nested structures."""
    if len(batch) == 0:
        return {}
    result = {}
    # Get all keys from all batch items
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())
    # Process each key
    for key in all_keys:
        # Skip items that don't have this key
        values = [item[key] for item in batch if key in item]
        if len(values) == 0:
            continue
        # Handle different types of values
        if isinstance(values[0], dict):
            # Create a new dictionary with batched values
            nested_dict = {}
            sub_keys = set()
            for v in values:
                sub_keys.update(v.keys())
            for sub_key in sub_keys:
                # Get values that have this sub_key
                sub_values = [v[sub_key] for v in values if sub_key in v]
                if len(sub_values) > 0:
                    if isinstance(sub_values[0], torch.Tensor):
                        # Stack tensors
                        try:
                            nested_dict[sub_key] = torch.stack(sub_values)
                        except:
                            # If can't stack (different sizes), keep as list
                            nested_dict[sub_key] = sub_values
                    else:
                        # Keep other types as lists
                        nested_dict[sub_key] = sub_values
            result[key] = nested_dict
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors when possible
            try:
                result[key] = torch.stack(values)
            except:
                # If tensors have different sizes, keep as list
                result[key] = values
        else:
            # For other types, just keep as list
            result[key] = values
    return result

import time

startime = time.time()
dataloader = create_multimodal_mind2web_dataloader(
    'Multimodal-Mind2Web/data',
    "train",
    tokenizer=tokenizer,
    visual_processor=visual_processor,
)
print(time.time() - startime)