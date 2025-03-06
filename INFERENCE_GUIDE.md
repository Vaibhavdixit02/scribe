# Cross-Modal Web Agent Inference Guide

This guide explains how to run inference and analyze results using the cross-modal web agent.

## Running Inference on Supercloud

1. First, modify `run_inference.sh` with your specific data paths:

```bash
# Edit these lines in run_inference.sh:
export MIND2WEB_DATA_PATH="/home/gridsan/vdixit/data/mind2web"  # Set to your actual data path
cd /home/gridsan/vdixit/scribe  # Set to your project directory
```

2. Next, submit the job using LLsub:

```bash
# Upload the necessary files
scp run_inference.sh run_inference.py visualize_results.py gridsan:/home/gridsan/your_username/scribe/

# Connect to Supercloud
ssh gridsan

# Submit the job with LLsub
cd /home/gridsan/your_username/scribe
llsub run_inference.sh
```

3. Check the job status:

```bash
# On Supercloud
llq
```

The SLURM directives in the script are already set up for the Volta GPUs on Supercloud.

## Downloading and Analyzing Results

Once your job is complete, use the `download_results.py` script to fetch and analyze the results:

```bash
# Replace JOB_ID with your actual LLsub job ID (like llsub_123456)
python download_results.py --job_id JOB_ID --user your_username --host gridsan --analyze
```

This will:
1. Check if the job is completed using LLsub commands
2. Download all result files from Supercloud
3. Generate visualizations and an HTML report
4. Open the report in your browser (if available)

The script can also monitor a running job and wait until it completes.

## Examining Results Manually

The downloaded results will include:

- `JOB_ID.log`: Log file from the job (LLsub format)
- `inference_results_TIMESTAMP.json`: Raw inference results from the model
- `analysis/prediction_report.html`: HTML report with predictions and screenshots
- `analysis/operation_types.png`: Graph of operation types detected

## Creating a Pickled Dataset for Faster Inference

The data processing step can be time-consuming. To speed up inference, you can pre-process the data and save it as a pickle file:

```bash
# Process data once and save as pickle
python create_pickled_dataset.py \
  --data_path /path/to/mind2web \
  --split test_domain \
  --num_examples 20 \
  --output_path processed_dataset.pkl
```

This creates a pickled dataset with already processed HTML and extracted visual features, which is much faster to load during inference.

## Customizing Inference

To customize the inference process, use these parameters:

```bash
# Using raw data
python run_inference.py \
  --model_path checkpoints/best_model \  # Path to your model
  --data_path $MIND2WEB_DATA_PATH \      # Path to Mind2Web data
  --split test_domain \                  # Which test split to use
  --num_examples 10 \                    # Number of examples to process
  --use_multi_gpu \                      # Use multiple GPUs
  --output_dir inference_results         # Where to save results

# Using pickled dataset (faster)
python run_inference.py \
  --model_path checkpoints/best_model \  # Path to your model
  --pickled_dataset processed_dataset.pkl \  # Use pre-processed data
  --use_multi_gpu \                      # Use multiple GPUs
  --output_dir inference_results         # Where to save results
```

## Troubleshooting

1. **Memory issues**: If you encounter CUDA out-of-memory errors, try:
   - Reducing batch size
   - Enabling gradient checkpointing
   - Using bfloat16 precision

2. **Missing dependencies**: Make sure to install all required packages:
   ```bash
   pip install accelerate einops tqdm matplotlib pillow
   ```

3. **Data path issues**: Verify the Mind2Web dataset structure:
   ```
   mind2web/
   ├── train/
   │   ├── task_*.json
   │   ├── raw_dump/
   │   └── screenshots/
   ├── test_domain/
   │   ├── task_*.json
   │   ├── raw_dump/
   │   └── screenshots/
   └── ...
   ```