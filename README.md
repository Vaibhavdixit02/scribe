# Cross-Modal Web Agent

This project develops a fine-tuned web agent that leverages both HTML DOM structure and visual screenshots to improve web navigation accuracy. By combining textual and visual information, we can create a more robust agent capable of handling ambiguous UI elements, understanding visual context, and achieving higher success rates on web automation tasks.

## Project Structure

```
scribe_agent/
├── configs/         # Configuration files
├── data/            # Data processing and loaders
├── models/          # Model architecture
├── utils/           # Helper functions
└── eval/            # Evaluation scripts
```

## Features

- Joint processing of HTML DOM and visual screenshots
- Cross-modal attention mechanism for aligning text and visual elements
- Element localization module to identify interactive elements
- Compatible with Mind2Web benchmark for evaluation
- Parameter-efficient fine-tuning with LoRA
- Support for distributed training

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

# Or install dependencies only
pip install -r requirements.txt
```

### Download Mind2Web Dataset

```bash
# Clone Mind2Web repository
git clone https://github.com/OSU-NLP-Group/Mind2Web.git

# Download and extract dataset files
mkdir -p data/mind2web
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

### Configuration

Modify the configuration in `configs/multimodal_config.yaml` to adjust model parameters, training settings, and data paths.

### Training Script

```bash
# Single GPU training
python scribe_agent/train.py --config scribe_agent/configs/multimodal_config.yaml

# Distributed training with multiple GPUs
torchrun --nproc_per_node=2 scribe_agent/train.py --config scribe_agent/configs/multimodal_config.yaml
```

### Training Monitoring

Training progress can be monitored via Weights & Biases or TensorBoard:

```bash
# Launch TensorBoard
tensorboard --logdir checkpoints/logs
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

## Demo

Launch the interactive demo to test the model on custom examples:

```bash
python scribe_agent/demo.py --model_path checkpoints/best_model
```

The demo provides a web interface where you can upload HTML content and a screenshot, specify a task, and see the model's predicted action.

## Expected Results

Performance improvements over text-only models:

- Element accuracy: +5-8% improvement
- Task success rate: +7-10% improvement on visually complex sites
- Significant gains on ambiguous UI elements and similar-looking components

## Integration with Scribe

The cross-modal agent can be integrated with ScribeAgent to enhance automation capabilities. The element localization module can improve precision on task recording and provide more robust web navigation.