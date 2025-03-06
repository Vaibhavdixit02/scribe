## Data Processing & Handling

The project includes sophisticated data processing pipelines to handle the complex multimodal inputs required for training:

### HTML Processing (`utils/html_processor.py`)

- **DOM Cleaning & Formatting**: Removes unnecessary HTML elements (scripts, styles) while preserving structural information
- **Node Identification**: Assigns unique node IDs to DOM elements for tracking
- **Attribute Pruning**: Retains only essential attributes to reduce input size
- **Context Window Management**: Implements smart truncation strategies to fit large DOMs within model context limits

### Visual Processing (`utils/visual_processor.py`)

- **Screenshot Handling**: Processes both full-page screenshots and element-level crops
- **Visual Feature Extraction**: Uses CLIP's vision encoder to extract semantic visual features
- **Element Bounding Box Integration**: Maps DOM elements to their visual regions in screenshots
- **Visual Context Formation**: Creates combined representations that connect visual and textual information

### Dataset Creation (`data/mind2web_dataset.py`)

- **Mind2Web Integration**: Adapts the Mind2Web dataset for multimodal training by adding visual components
- **Example Construction**: Each training example includes:
  - Task description (e.g., "Book a flight from New York to San Francisco")
  - URL and DOM structure
  - Screenshot with element bounding boxes
  - Target element and action (click, type, select)
- **Efficiency Optimizations**: Implements pickling for preprocessed data to avoid redundant computation
- **Data Augmentation**: Supports targeted data augmentation to handle DOM and visual variations

### Data Pipeline Scripts

The repository includes several scripts for data handling:

- `data_creator.py`: Creates and pickles the dataset for efficient training
- `run_inference.py`: Handles optimized data loading during inference
- `dataset_creator.sh`: Shell script for running dataset creation on compute clusters

Example usage pattern:

```python
# From data_creator.py
dataloader = create_multimodal_mind2web_dataloader(
    'Multimodal-Mind2Web/data',
    "train",
    tokenizer=tokenizer,
    visual_processor=visual_processor,
)

# Pickle for reuse
save_object(dataset, 'dataset_full.pkl')
```# Cross-Modal Web Agent

This project is an experimental approach to develop a web agent that leverages both HTML DOM structure and visual screenshots to improve web navigation accuracy. Inspired by my work on a Chrome extension for IPO market information in India and experience with Gemini's 1M context window for multimodal applications, this project attempts to combine textual and visual information to create a more robust agent capable of handling ambiguous UI elements and understanding visual context.

## The Importance of Multimodality in Browser Automation

Traditional browser automation relies primarily on DOM selectors, XPaths, or text-based identification, which often fails when:

1. **UI Ambiguity**: Multiple elements share similar text or attributes (e.g., multiple "Add to Cart" buttons)
2. **Dynamic Content**: Elements change position or attributes between page loads
3. **Visual Confirmation**: Tasks that require visual verification (e.g., "select the red shirt")
4. **Shadow DOM/Iframes**: Complex modern web apps with encapsulated components

Multimodal agents address these challenges by integrating:
- **Textual Understanding**: Processing DOM structure, element attributes, and content
- **Visual Understanding**: Recognizing UI elements by appearance, layout, and visual context
- **Cross-modal Reasoning**: Connecting what an element "looks like" with what it "does"

This approach is particularly valuable for:
- E-commerce automation (product selection based on appearance)
- Form filling with complex validation
- Web testing that requires visual confirmation
- Accessibility-focused automation

> **Note:** This project is currently in the experimental stage and fine-tuning has not yet been successfully completed. The architecture and approach described below represent the design goals rather than proven results.

## Project Structure

```
scribe_agent/
├── configs/         # Configuration files for training and inference
├── data/            # Data processing and dataset loaders
├── models/          # Model architecture definitions
├── utils/           # HTML and visual processing utilities
└── eval/            # Evaluation scripts
```

## Key Features

- **Multimodal Understanding**: Joint processing of HTML DOM and visual screenshots to provide a richer understanding of web interfaces
- **Cross-modal Attention**: Custom attention mechanism that aligns text tokens with visual regions for better element disambiguation
- **Element Localization**: Specialized module to identify interactive elements by combining visual and textual cues
- **Mind2Web Integration**: Compatible with the Mind2Web benchmark for standardized evaluation
- **Parameter-Efficient Design**: LoRA-based fine-tuning approach to minimize computational requirements
- **Distributed Architecture**: Support for distributed training and inference via DeepSpeed ZeRO-3 and Accelerate library
- **Memory Optimization**: Various techniques to manage the substantial memory requirements of large models

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with 24GB+ VRAM
- 150GB disk space (for dataset and model checkpoints)

### Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/your-organization/cross-modal-web-agent.git
cd cross-modal-web-agent

# Install the package in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Download Mind2Web Dataset

```bash
# Create data directory
mkdir -p data/mind2web

# Download and extract dataset files
cd data/mind2web
wget https://mind2web.oss-us-west-1.aliyuncs.com/data/train.zip
wget https://mind2web.oss-us-west-1.aliyuncs.com/data/test_task.zip
wget https://mind2web.oss-us-west-1.aliyuncs.com/data/test_website.zip
wget https://mind2web.oss-us-west-1.aliyuncs.com/data/test_domain.zip
unzip train.zip
unzip test_task.zip
unzip test_website.zip
unzip test_domain.zip
cd ../..
```

## Training

### Environment Setup

The training process requires significant computational resources:

- At least one GPU with 32GB+ VRAM (V100/A100 recommended)
- Alternatively, multiple GPUs with DeepSpeed ZeRO-3 for distributed training
- 150GB+ disk space for dataset and model checkpoints

### Configuration

Modify the configuration in `scribe_agent/configs/multimodal_config.yaml` to adjust model parameters, training settings, and data paths:

```yaml
# Key configuration options
model:
  text_model_name: "Qwen/Qwen2-7B"
  vision_model_name: "openai/clip-vit-base-patch32"
  use_lora: true
  lora_rank: 16
  lora_alpha: 32

training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
```

### Training Script

```bash
# Single GPU training (will likely run out of memory)
python scribe_agent/train.py --config scribe_agent/configs/multimodal_config.yaml

# Recommended: Distributed training with DeepSpeed ZeRO-3
accelerate launch --config_file default_config.yaml scribe_agent/train.py --config scribe_agent/configs/multimodal_config.yaml
```

### Memory Management

The current training script includes several strategies to manage the substantial memory requirements:

- DeepSpeed ZeRO-3 for parameter offloading and partitioning
- Gradient checkpointing to reduce activation memory
- 8-bit model quantization for the base models
- Carefully tuned batch size and gradient accumulation steps

### Training Monitoring

Training progress can be monitored via Weights & Biases or TensorBoard:

```bash
# Launch TensorBoard
tensorboard --logdir checkpoints/logs
```

> **Current Status**: The training pipeline is implemented but encountering challenges with memory management and optimization stability. Work is ongoing to resolve these issues.

## Inference

Run inference using the provided scripts:

```bash
# Using DeepSpeed ZeRO-3 with distributed inference
bash run_inference.sh

# Or run directly with Python
python run_inference.py --model_path checkpoints/best_model --pickled_dataset dataset.pkl --output_dir inference_results
```

For faster inference, you can create a pickled dataset:

```bash
# Process a dataset once and save as pickle
python data_creator.py
```

## Evaluation

Evaluate the model on the Mind2Web test splits:

```bash
# Evaluate on the domain generalization test split
python scribe_agent/evaluate.py --model_path checkpoints/best_model --split test_domain --output_file results/domain_results.json

# Evaluate on the task generalization test split
python scribe_agent/evaluate.py --model_path checkpoints/best_model --split test_task --output_file results/task_results.json

# Evaluate on the website generalization test split
python scribe_agent/evaluate.py --model_path checkpoints/best_model --split test_website --output_file results/website_results.json
```

## Target Results

Expected performance improvements over text-only models (once fine-tuning is successful):

- Element accuracy: +5-8% improvement target
- Task success rate: +7-10% improvement target on visually complex sites
- Potential gains on ambiguous UI elements and similar-looking components

These are aspirational targets based on theoretical architecture advantages rather than current achieved results.

## Inference Guide

For detailed inference instructions, see [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md).

## Architectural Design & Technical Details

The cross-modal web agent employs a specialized architecture designed to bridge textual and visual understanding:

### Model Components

1. **Text Foundation**: Qwen2-7B serves as the base language model for processing HTML DOM structures and user instructions
   - Selected for its strong instruction-following capabilities and context length support
   - Uses 8-bit quantization to reduce memory footprint

2. **Visual Processing**: CLIP vision encoder (ViT-B/32) processes webpage screenshots
   - Extracts visual features from both full screenshots and individual UI elements
   - Provides complementary information when text descriptions are ambiguous

3. **Cross-Modal Integration**: Custom cross-attention mechanism aligns DOM elements with visual features
   - Multi-head attention from text tokens to visual tokens enables the model to "look at" relevant parts of the image
   - Bidirectional information flow helps resolve ambiguities in either modality

4. **Element Localization Module**: Specialized classifier head predicts which HTML element to interact with
   - Uses enhanced representations from cross-attention to make more accurate predictions
   - Designed to handle visually similar but functionally different elements

### Performance Optimizations

- **Memory Efficiency**:
  - DeepSpeed ZeRO-3 for distributed training across multiple GPUs
  - 8-bit model quantization to reduce memory footprint
  - Gradient checkpointing to trade computation for memory

- **Training Efficiency**:
  - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
  - Only trains approximately 0.1% of the parameters compared to full fine-tuning
  - Focuses adaptation on key weight matrices (query, key, value projections)

- **Inference Optimizations**:
  - Pickled dataset approach to avoid repeated processing of HTML and screenshots
  - Partial model offloading to manage memory constraints
  - Accelerate library integration for simplified distributed inference

### Current Challenges

- Fine-tuning pipeline has encountered memory and optimization issues
- Cross-modal alignment requires significant computational resources
- Balancing parameter efficiency with model expressivity remains challenging

The implementation includes careful input processing to handle the complex structure of web pages, with truncation strategies to fit within model context windows.