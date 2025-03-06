import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm

from data.mind2web_dataset import create_mind2web_dataloader
from models.cross_modal_model import CrossModalWebAgent
from utils.visual_processor import VisualProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a cross-modal web agent")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/multimodal_config.yaml", 
                      help="Path to configuration file")
    parser.add_argument("--split", type=str, default="test_domain", 
                      choices=["test_task", "test_website", "test_domain"],
                      help="Test split to evaluate on")
    parser.add_argument("--output_file", type=str, default=None,
                      help="Path to save evaluation results")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_element_match(predicted_node_id, target_node_id):
    """Check if predicted element matches target element."""
    return predicted_node_id == target_node_id

def calculate_f1(predicted_operation, target_operation):
    """Calculate F1 score for operation prediction."""
    # Simplification: For click operations, return 1.0 if both are clicks
    # For type operations, check if the value is correct
    if predicted_operation["op"] != target_operation["op"]:
        return 0.0
    
    if predicted_operation["op"] == "CLICK":
        return 1.0
    elif predicted_operation["op"] == "TYPE":
        # For type operations, check if the input value is correct
        pred_value = predicted_operation.get("value", "")
        target_value = target_operation.get("value", "")
        
        # Calculate string similarity
        if not pred_value or not target_value:
            return 0.0
            
        # Simple exact match for now
        return 1.0 if pred_value == target_value else 0.0
    
    return 0.0

def parse_model_output(output_text):
    """Parse the model output to extract operation and target element."""
    if "Click the element with node ID" in output_text:
        # Extract node ID for click operation
        parts = output_text.split("Click the element with node ID")
        if len(parts) > 1:
            node_id = parts[1].strip()
            return {
                "node_id": node_id,
                "operation": {"op": "CLICK"}
            }
    elif "Type '" in output_text and "' in element with node ID" in output_text:
        # Extract node ID and value for type operation
        parts = output_text.split("' in element with node ID")
        if len(parts) > 1:
            node_id = parts[1].strip()
            value_part = parts[0].split("Type '")
            if len(value_part) > 1:
                value = value_part[1].strip()
                return {
                    "node_id": node_id,
                    "operation": {"op": "TYPE", "value": value}
                }
    
    # Fallback: Try to find any node ID in the text
    import re
    node_match = re.search(r'node ID (\d+)', output_text)
    if node_match:
        node_id = node_match.group(1)
        # Assume CLICK if operation not clearly specified
        return {
            "node_id": node_id,
            "operation": {"op": "CLICK"}
        }
    
    # Could not parse output
    return None

def generate_prediction(model, tokenizer, sample, max_length=200):
    """Generate prediction from the model."""
    # Prepare input
    prompt = f"Objective: {sample['task_description']}\nURL: {sample['url']}\n"
    input_text = prompt + "\n" + sample["processed_html"]
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=32768,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Add visual features if available
    visual_features = None
    if "visual_features" in sample:
        visual_features = sample["visual_features"]
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            visual_features=visual_features,
            max_length=inputs["input_ids"].shape[1] + max_length,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    return generated_text

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = CrossModalWebAgent.from_pretrained(
        args.model_path,
        text_model_name=config["model"]["text_model_name"],
        vision_model_name=config["model"]["vision_model_name"]
    ).to(device)
    model.eval()
    
    # Initialize visual processor
    visual_processor = VisualProcessor(
        vision_model_name=config["model"]["vision_model_name"]
    )
    
    # Create dataloader
    test_dataloader = create_mind2web_dataloader(
        data_dir=config["data"]["data_dir"],
        split=args.split,
        tokenizer=None,  # We'll handle tokenization in the evaluation loop
        batch_size=1,
        shuffle=False,
        num_workers=4,
        max_length=config["model"]["max_length"]
    )
    
    # Metrics
    metrics = {
        'element_accuracy': 0,
        'action_f1': 0,
        'step_success': 0,
        'visual_localization': 0,
        'total': 0
    }
    
    # Collect detailed results
    detailed_results = []
    
    # Evaluate
    for batch in tqdm(test_dataloader, desc=f"Evaluating on {args.split}"):
        # Generate prediction
        sample = batch  # Since batch_size=1
        generated_text = generate_prediction(model, tokenizer, sample)
        
        # Parse prediction
        parsed_prediction = parse_model_output(generated_text)
        
        # Get target
        target_node_id = sample["target_node_id"]
        target_operation = sample["target_element"]["operation"]
        
        # Skip if prediction couldn't be parsed
        if not parsed_prediction:
            detailed_results.append({
                "task_id": sample["task_id"],
                "step_id": sample["step_id"],
                "generated_text": generated_text,
                "parsed_prediction": None,
                "target_node_id": target_node_id,
                "target_operation": target_operation,
                "element_match": False,
                "action_f1": 0.0,
                "step_success": False
            })
            metrics['total'] += 1
            continue
        
        # Calculate metrics
        element_match = calculate_element_match(
            parsed_prediction["node_id"], 
            target_node_id
        )
        
        action_f1 = calculate_f1(
            parsed_prediction["operation"],
            target_operation
        )
        
        step_success = element_match and action_f1 > 0.5
        
        # Update metrics
        metrics['element_accuracy'] += int(element_match)
        metrics['action_f1'] += action_f1
        metrics['step_success'] += int(step_success)
        metrics['total'] += 1
        
        # Store detailed result
        detailed_results.append({
            "task_id": sample["task_id"],
            "step_id": sample["step_id"],
            "generated_text": generated_text,
            "parsed_prediction": parsed_prediction,
            "target_node_id": target_node_id,
            "target_operation": target_operation,
            "element_match": element_match,
            "action_f1": action_f1,
            "step_success": step_success
        })
    
    # Calculate average metrics
    for key in metrics:
        if key != 'total':
            metrics[key] = metrics[key] / metrics['total'] if metrics['total'] > 0 else 0
    
    # Print results
    print(f"=== Evaluation Results on {args.split} ===")
    print(f"Element Accuracy: {metrics['element_accuracy']:.4f}")
    print(f"Action F1: {metrics['action_f1']:.4f}")
    print(f"Step Success Rate: {metrics['step_success']:.4f}")
    print(f"Total Examples: {metrics['total']}")
    
    # Save results
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(args.output_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "detailed_results": detailed_results
            }, f, indent=2)
        
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()