import os
import argparse
import torch
from transformers import AutoTokenizer
import gradio as gr

from models.cross_modal_model import CrossModalWebAgent
from utils.visual_processor import VisualProcessor
from utils.html_processor import process_html

def parse_args():
    parser = argparse.ArgumentParser(description="Demo for cross-modal web agent")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to model checkpoint")
    return parser.parse_args()

def process_example(html_content, screenshot, task_description, url):
    """Process a single example for the model."""
    # Process HTML
    processed_html, _ = process_html(html_content)
    
    # Extract visual features
    visual_features = visual_processor.extract_visual_features(
        screenshot,
        {}  # We don't have bounding boxes in demo mode
    )
    
    # Create prompt
    prompt = f"Objective: {task_description}\nURL: {url}\n"
    input_text = prompt + "\n" + processed_html
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=model.text_model.config.max_position_embeddings,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            visual_features=visual_features,
            max_new_tokens=100
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    return generated_text

def demo_interface(html_content, screenshot, task_description, url):
    # Process example
    prediction = process_example(html_content, screenshot, task_description, url)
    
    # Create side-by-side view
    result = {
        "Task": task_description,
        "URL": url,
        "Model Prediction": prediction
    }
    
    return result

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model
    model = CrossModalWebAgent.from_pretrained(args.model_path).to(device)
    model.eval()
    
    # Initialize visual processor
    visual_processor = VisualProcessor()
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Cross-Modal Web Agent Demo")
        
        with gr.Row():
            with gr.Column():
                html_input = gr.Textbox(
                    label="HTML Content",
                    placeholder="Paste HTML content here...",
                    lines=10
                )
                
                screenshot_input = gr.Image(
                    label="Screenshot",
                    type="numpy"
                )
                
            with gr.Column():
                task_input = gr.Textbox(
                    label="Task Description",
                    placeholder="e.g., Book a flight from New York to San Francisco"
                )
                
                url_input = gr.Textbox(
                    label="URL",
                    placeholder="e.g., https://example.com"
                )
                
                submit_button = gr.Button("Predict Action")
        
        output_display = gr.JSON(label="Prediction Result")
        
        submit_button.click(
            fn=demo_interface,
            inputs=[html_input, screenshot_input, task_input, url_input],
            outputs=output_display
        )
    
    # Launch demo
    demo.launch(share=True)