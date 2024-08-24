#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_bin_path> <eval_data_path> <stockfish_path>"
    exit 1
fi

# Assign arguments to variables
model_bin_path="$1"
data_path="$2"
stockfish_path="$3"

# Define the HF model path
model_hf_path="${model_bin_path%/*}/rook_gpt2_124M_hf"

# Define the aggregate log file
aggregate_log="aggregate_output.log"

# Convert the .bin model to HF format
echo "Converting model to HF format..."
python export_hf.py -i "$model_bin_path" -o "$model_hf_path" --spin False

if [ $? -ne 0 ]; then
    echo "Error: Model conversion failed. Exiting."
    exit 1
fi

echo "Model converted successfully. HF model path: $model_hf_path"
echo "Running 4 evals in parallel. This can take 20-30 minutes."
echo ""

# Check Multi-GPU available
gpu_count=$(nvidia-smi -L | wc -1)
if [ $gpu_count -gt 1 ]; then
    cuda_device=1
else
    cuda_device=0
fi
echo "Number of GPUs detected: $gpu_count."

# Function to run a command and log its output
run_command() {
    local command="$1"
    local temp_log="$2"
    local header="$3"
    echo "Running: $command"
    
    # Run the command and capture output to a temporary file
    $command > "$temp_log" 2>&1 &
    local pid=$!
    echo "PID: $!"
}

# Create temporary log files
temp_log1=$(mktemp)
temp_log2=$(mktemp)
temp_log3=$(mktemp)
temp_log4=$(mktemp)

# Run the four Python commands in parallel with the provided arguments
run_command "python3 rook_accuracy.py -m $model_hf_path -d $data_path -g" "$temp_log1" "ROOK Validation Accuracy Evaluation"
run_command "CUDA_VISIBLE_DEVICES=$cuda_device python3 rook_bb-cio.py -m $model_hf_path -g" "$temp_log2" "ROOK BIG-bench Checkmate In One Evaluation"
run_command "python3 rook_selfplay.py -m $model_hf_path" "$temp_log3" "ROOK Self-play Evaluation"
run_command "python3 rook_vs_stockfish.py -m $model_hf_path -g -p $stockfish_path" "$temp_log4" "ROOK vs Stockfish Evaluation"

# Wait for all background processes to finish
wait

# Function to append a log with a header to the aggregate log
append_log() {
    local header="$1"
    local log_file="$2"
    local command="$3"
    {
        echo "===== $header ====="
        echo "Command: $command"
	tail -n 10 "$log_file"
        echo ""  # Add a blank line for separation
    } >> "$aggregate_log"
}

# Clear the aggregate log if it exists
> "$aggregate_log"

# Append each log to the aggregate log with headers
append_log "ROOK Validation Accuracy Evaluation (rook_accuracy.py -m $model_hf_path -d $data_path -g)" "$temp_log1"
append_log "ROOK BIG-bench Checkmate In One Evaluation (rook_bb-cio.py -m $model_hf_path -g)" "$temp_log2"
append_log "ROOK Self-play Evaluation (rook_selfplay.py -m $model_hf_path)" "$temp_log3"
append_log "ROOK vs Stockfish Evaluation (rook_vs_stockfish.py -m $model_hf_path -g -p $stockfish_path)" "$temp_log4"

# Remove temporary log files
rm "$temp_log1" "$temp_log2" "$temp_log3" "$temp_log4"

echo "All commands have completed. Aggregate output is in $aggregate_log"
