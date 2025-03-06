import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import pickle
import time

# Set environment variables before imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
# Import from our package
from scribe_agent.models.cross_modal_model import CrossModalWebAgent
from scribe_agent.utils.visual_processor import VisualProcessor
from scribe_agent.utils.html_processor import process_html
from accelerate import Accelerator

TEXT_MODEL = "../Qwen2.5-7B-Instruct"
VISION_MODEL = "clip-vit-base-patch32"

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with cross-modal web agent")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model",
                        help="Path to the model checkpoint")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to Mind2Web dataset")
    parser.add_argument("--pickled_dataset", type=str, default='dataset.pkl',
                        help="Path to pickled dataset file (faster than processing from raw data)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test_task", "test_website", "test_domain"],
                        help="Dataset split to use")
    parser.add_argument("--task_id", type=str, default=None,
                        help="Specific task ID to evaluate (optional)")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of examples to process")
    parser.add_argument("--use_multi_gpu", action="store_true",
                        help="Use multiple GPUs via Accelerate")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pickled_dataset is None and args.data_path is None:
        parser.error("Either --pickled_dataset or --data_path must be provided")
        
    return args

def setup_model_and_tokenizer(args, accelerator=None):
    """Setup model and tokenizer with proper device handling for DeepSpeed ZeRO-3"""
    print(f"Setting up model and tokenizer...")
    
    # Initialize tokenizer without internet access
    if os.path.exists(args.model_path):
        print(f"Loading tokenizer from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    else:
        print("Model path not found, using Qwen2-7B tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, local_files_only=True)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model with proper config for DeepSpeed ZeRO-3
    # Important: When using DeepSpeed ZeRO-3, don't use device_map="auto"
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        # Don't specify device_map when using DeepSpeed
        model = CrossModalWebAgent.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,  # Use fp16 for efficiency
            local_files_only=True
        )
    else:
        print("Model path not found, initializing new model")
        model = CrossModalWebAgent(
            text_model_name=TEXT_MODEL,
            vision_model_name=VISION_MODEL,
            use_lora=False
        )
    
    print(f"Model initialized, waiting for DeepSpeed preparation...")
    return model, tokenizer

def setup_visual_processor():
    """Setup visual processor"""
    return VisualProcessor(vision_model_name=VISION_MODEL)

def find_tasks(args):
    """Find task files in the dataset split or load from pickle."""
    # First check if we should use pickled dataset
    if hasattr(args, 'pickled_dataset') and args.pickled_dataset:
        print(f"Loading data from pickled dataset: {args.pickled_dataset}")
        try:
            with open(args.pickled_dataset, 'rb') as f:
                data = pickle.load(f)
            print(f"Successfully loaded pickled dataset with {len(data)} tasks")
            return data
        except Exception as e:
            print(f"Error loading pickled dataset: {e}")
            print("Falling back to regular task loading...")
            return []  # Return empty list or implement fallback
    return []  # Implement your fallback loading logic here

def custom_collate_fn(batch):
    """
    Simplified collate function to handle complex nested structures.
    Optimized for DeepSpeed compatibility.
    """
    if len(batch) == 0:
        return {}
    
    result = {}
    # Get all keys from the first batch item (assuming consistent keys)
    all_keys = batch[0].keys()
    
    # Process each key
    for key in all_keys:
        values = [item[key] for item in batch if key in item]
        if len(values) == 0:
            continue
            
        # Handle different types of values
        if isinstance(values[0], torch.Tensor):
            # Try to stack tensors when possible
            try:
                result[key] = torch.stack(values)
            except:
                # If tensors have different sizes, keep as list
                result[key] = values
        elif isinstance(values[0], dict):
            # Create a new dictionary with batched values
            nested_dict = {}
            sub_keys = values[0].keys()
            
            for sub_key in sub_keys:
                # Get values that have this sub_key
                sub_values = [v[sub_key] for v in values if sub_key in v]
                if len(sub_values) > 0:
                    if isinstance(sub_values[0], torch.Tensor):
                        # Stack tensors when possible
                        try:
                            nested_dict[sub_key] = torch.stack(sub_values)
                        except:
                            nested_dict[sub_key] = sub_values
                    else:
                        nested_dict[sub_key] = sub_values
            result[key] = nested_dict
        else:
            # For other types, just keep as list
            result[key] = values
    
    return result

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing Accelerator...")
    # Initialize Accelerator first (before any model loading)
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with=None,
        project_dir=args.output_dir
    )
    
    # Log accelerator configuration
    print(f"Accelerator initialized with:")
    print(f"  - Device: {accelerator.device}")
    print(f"  - Distributed type: {accelerator.distributed_type}")
    print(f"  - Process index: {accelerator.process_index} of {accelerator.num_processes}")
    print(f"  - Local process index: {accelerator.local_process_index}")
    
    # Setup model, tokenizer, and visual processor
    model, tokenizer = setup_model_and_tokenizer(args, accelerator)
    visual_processor = setup_visual_processor()
    
    # Find task files or load pickled dataset
    dataset = find_tasks(args)
    if not dataset:
        print("No dataset loaded, exiting.")
        return
    
    # Define split sizes
    train_ratio = 0.9
    val_ratio = 1 - train_ratio
    train_size = len(dataset) - 1
    val_size = 1
    
    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training examples, {val_size} validation examples")
    
    # Create DataLoaders with a simple wrapper to make sure they're DeepSpeed compatible
    # Use smaller batch size for inference to avoid OOM
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # Keep batch size small for inference
        shuffle=True, 
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # IMPORTANT: Prepare model and dataloaders with accelerator
    # This is where DeepSpeed ZeRO-3 sharding happens
    print("Preparing model with DeepSpeed...")

    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process each batch for inference
    all_results = []
    
    # Get a validation batch for testing
    try:
        val_iter = iter(val_loader)
        batch = next(val_iter)
        print("Successfully loaded validation batch")
    except Exception as e:
        print(f"Error getting validation batch: {e}")
        return
    
    # Run inference on the batch with timing
    print("Running inference...")
    start_time = time.time()
    
    # Ensure we're in no_grad context for inference
    with torch.no_grad():
        try:
            # Forward pass with the prepared model
            output = model(
                input_ids=batch.get('input_ids', None),
                attention_mask=batch.get('attention_mask', None),
                visual_features=batch.get('visual_features', None),
                labels=batch.get('labels', None)
            )
            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.2f} seconds")
            
            # Try to gather outputs if in distributed setup
            if accelerator.num_processes > 1:
                print("Gathering outputs from all processes...")
                output = accelerator.gather(output)
            
            all_results.append({
                'output': output,
                'inference_time': inference_time
            })
        except Exception as e:
            print(f"Error during inference: {e}")
    
    # Save results (only on main process)
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"inference_results_{timestamp}.json")
        
        # Convert outputs to serializable format
        serializable_results = []
        for result in all_results:
            # Handle various output types
            if hasattr(result['output'], 'to_dict'):
                output_dict = result['output'].to_dict()
            elif isinstance(result['output'], dict):
                # Convert tensors to lists
                output_dict = {}
                for k, v in result['output'].items():
                    if isinstance(v, torch.Tensor):
                        output_dict[k] = v.detach().cpu().tolist()
                    else:
                        output_dict[k] = v
            else:
                output_dict = {'raw_output': str(result['output'])}
            
            serializable_results.append({
                'output': output_dict,
                'inference_time': result['inference_time']
            })
        
        # Save to JSON
        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    print("All processes completed")

if __name__ == "__main__":
    main()