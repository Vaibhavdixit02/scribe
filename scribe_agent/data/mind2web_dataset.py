import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from ..utils.html_processor import process_html
from ..utils.visual_processor import VisualProcessor

class MultimodalMind2WebDataset(Dataset):
    def __init__(self, 
                 data_dir,
                 split="train",
                 tokenizer=None,
                 max_length=32768,
                 visual_processor=None):
        """
        Multimodal-Mind2Web dataset for multimodal web agent training.
        
        Args:
            data_dir: Path to the Multimodal-Mind2Web dataset
            split: One of 'train', 'test_task', 'test_website', 'test_domain'
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            visual_processor: Image processor for screenshots
        """
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Initialize visual processor if not provided
        if visual_processor is None:
            self.visual_processor = VisualProcessor()
        else:
            self.visual_processor = visual_processor
            
        # Load data
        self.examples = []
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess Multimodal-Mind2Web data from parquet files."""
        
        # Determine data path based on split
        if self.split == "train":
            parquet_files = [f for f in os.listdir(self.data_dir) if f.startswith("train-") and f.endswith(".parquet")]
            prefix = "train"
        elif self.split == "test_task":
            parquet_files = [f for f in os.listdir(self.data_dir) if f.startswith("test_task-") and f.endswith(".parquet")]
            prefix = "test_task"
        elif self.split == "test_website":
            parquet_files = [f for f in os.listdir(self.data_dir) if f.startswith("test_website-") and f.endswith(".parquet")]
            prefix = "test_website"
        elif self.split == "test_domain":
            parquet_files = [f for f in os.listdir(self.data_dir) if f.startswith("test_domain-") and f.endswith(".parquet")]
            prefix = "test_domain"
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Sort files to ensure deterministic order
        parquet_files.sort()
        
        for parquet_file in parquet_files:
            print(parquet_file)
            file_path = os.path.join(self.data_dir, parquet_file)
                
            # Read parquet file into pandas DataFrame
            df = pd.read_parquet(file_path)
                
            for idx, row in df.iterrows():
                # Extract task info
                task_id = row.get("annotation_id")
                task_description = row.get("confirmed_task")
                action_id = row.get("action_uid")
                
                # Skip rows without positive candidates
                if "pos_candidates" not in row or len(row["pos_candidates"]) == 0:
                    continue
                
                # Get screenshot path
                screenshot_path = row.get("screenshot")
                
                # # Make sure the path is absolute if it's relative
                # if screenshot_path and not os.path.isabs(screenshot_path):
                #     screenshot_path = os.path.join(self.data_dir, screenshot_path)
                
                # # Check if screenshot exists
                # if not screenshot_path or not os.path.exists(screenshot_path):
                #     continue
                
                # Use cleaned_html if available, otherwise use raw_html
                html_content = None
                if "cleaned_html" in row and row["cleaned_html"]:
                    html_content = row["cleaned_html"]
                elif "raw_html" in row and row["raw_html"]:
                    html_content = row["raw_html"]
                
                if not html_content:
                    continue
                
                # Process HTML
                processed_html, node_map = process_html(html_content)
                
                # Extract element bounding boxes
                element_bboxes = {}
                target_element = None
                
                # Handle pos_candidates (will be a string that needs to be parsed)
                pos_candidates = row["pos_candidates"]
                if isinstance(pos_candidates, str):
                    try:
                        pos_candidates = json.loads(pos_candidates)
                    except Exception as e:
                        print(f"Error parsing pos_candidates: {e}")
                        continue
                # Handle the case where pos_candidates might be a list of strings
                if isinstance(pos_candidates, list) or isinstance(pos_candidates, np.ndarray):
                    parsed_candidates = []
                    for candidate in pos_candidates:
                        if isinstance(candidate, str):
                            try:
                                parsed_candidate = json.loads(candidate)
                                parsed_candidates.append(parsed_candidate)
                            except Exception as e:
                                print(f"Error parsing candidate: {e}")
                                continue
                        else:
                            parsed_candidates.append(candidate)
                    pos_candidates = parsed_candidates
        
                # If pos_candidates is still a string after parsing, try one more time
                if isinstance(pos_candidates, str):
                    try:
                        pos_candidates = json.loads(pos_candidates)
                    except:
                        continue
        
                for candidate in pos_candidates:
                    # Find node ID in our mapping
                    # First, check if backend_node_id is directly in the candidate
                    backend_node_id = candidate.get("backend_node_id")
                    
                    # If not, it might be in the attributes field as a nested JSON string
                    if not backend_node_id and "attributes" in candidate:
                        attributes = candidate["attributes"]
                        if isinstance(attributes, str):
                            try:
                                attr_dict = json.loads(attributes)
                                backend_node_id = attr_dict.get("backend_node_id")
                            except:
                                pass
                    
                    if backend_node_id:
                        # Find corresponding node ID in our mapping
                        for node_id, mapped_id in node_map.items():
                            if mapped_id == backend_node_id:
                                # Extract bounding box info if available
                                # Try direct bbox first
                                if "bbox" in candidate:
                                    element_bboxes[node_id] = candidate["bbox"]
                                # If not directly available, check if it's in attributes
                                elif "attributes" in candidate:
                                    attributes = candidate["attributes"]
                                    if isinstance(attributes, str):
                                        try:
                                            attr_dict = json.loads(attributes)
                                            if "bounding_box_rect" in attr_dict:
                                                element_bboxes[node_id] = attr_dict["bounding_box_rect"]
                                        except:
                                            pass
                                    # Mark the first positive candidate as target
                                    if target_element is None:
                                        # Handle operation (might be a string that needs to be parsed)
                                        operation = row["operation"]
                                        if isinstance(operation, str):
                                            try:
                                                operation = json.loads(operation)
                                            except:
                                                continue
                                                
                                        # If operation is still a string after first parsing, try again
                                        if isinstance(operation, str):
                                            try:
                                                operation = json.loads(operation)
                                            except:
                                                continue
                                                
                                        target_element = {
                                            "node_id": node_id,
                                            "operation": operation
                                        }
                                break
                
                # Skip if we couldn't map any elements
                if not element_bboxes or target_element is None:
                    continue
                
                # Add the human-readable representation if available
                target_action_repr = row.get("target_action_reprs", "")
                
                # Create example
                example = {
                    "task_id": task_id,
                    "step_id": action_id,
                    "task_description": task_description,
                    "url": row.get("url", ""),
                    "screenshot_path": screenshot_path,
                    "processed_html": processed_html,
                    "element_bboxes": element_bboxes,
                    "target_element": target_element,
                    "website": row.get("website", ""),
                    "domain": row.get("domain", ""),
                    "subdomain": row.get("subdomain", ""),
                    "target_action_repr": target_action_repr
                }
                self.examples.append(example)
        
        return self.examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]

        # Extract visual features for the screenshot
        visual_features = self.visual_processor.extract_visual_features(
            example["screenshot_path"],
            example["element_bboxes"]
        )
        
        # Create input prompt
        prompt = f"Objective: {example['task_description']}\nURL: {example['url']}\n"
        
        # Create target output based on operation type
        target_node_id = example["target_element"]["node_id"]
        operation = example["target_element"]["operation"]
        action_type = operation["op"]
        
        # Use the human-readable action representation if available
        if example.get("target_action_repr"):
            target = example["target_action_repr"]
        else:
            # Otherwise construct it programmatically
            if action_type == "CLICK":
                target = f"Click the element with node ID {target_node_id}"
            elif action_type == "TYPE":
                value = operation.get("value", "")
                target = f"Type '{value}' in element with node ID {target_node_id}"
            elif action_type == "SELECT":
                value = operation.get("value", "")
                target = f"Select '{value}' in element with node ID {target_node_id}"
            else:
                # Handle other operations as needed
                target = f"Perform {action_type} on element with node ID {target_node_id}"
        
        # Tokenize input (only if tokenizer is provided)
        if self.tokenizer:
            # Combine prompt with HTML
            input_text = prompt + "\n" + example["processed_html"]
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenize target
            targets = self.tokenizer(
                target,
                max_length=100,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": targets["input_ids"].squeeze(),
                "visual_features": visual_features,
                "task_id": example["task_id"],
                "step_id": example["step_id"],
                "target_node_id": target_node_id
            }
        
        # Return raw data if no tokenizer
        return {
            "prompt": prompt,
            "html": example["processed_html"],
            "visual_features": visual_features,
            "target": target,
            "task_id": example["task_id"],
            "step_id": example["step_id"],
            "target_node_id": target_node_id
        }

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def create_multimodal_mind2web_dataloader(data_dir, split, tokenizer=None, visual_processor=None, batch_size=4, 
                                         shuffle=True, num_workers=4, max_length=32768):
    """Create a dataloader for Multimodal-Mind2Web dataset."""
    dataset = MultimodalMind2WebDataset(
        data_dir=data_dir,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        visual_processor=visual_processor
    )

    save_object(dataset, 'dataset_full.pkl')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader