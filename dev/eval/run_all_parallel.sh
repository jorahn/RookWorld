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
python export_hf.py -i "$model_bin_path" -o "$model_hf_path" -s 0

if [ $? -ne 0 ]; then
    echo "Error: Model conversion failed. Exiting."
    exit 1
fi

echo "Model converted successfully. HF model path: $model_hf_path"

# Function to run a command and log its output
run_command() {
    local command="$1"
    local temp_log="$2"
    local header="$3"
    echo "Running: $command"
    
    # Run the command and capture output to a temporary file
    $command > "$temp_log" 2>&1 &
    echo "PID: $!"
}

# Create temporary log files
temp_log1=$(mktemp)
temp_log2=$(mktemp)
temp_log3=$(mktemp)
temp_log4=$(mktemp)

# Run the four Python commands in parallel with the provided arguments
run_command "python3 rook_accuracy.py -m $model_hf_path -d $data_path -g" "$temp_log1" "ROOK Accuracy Evaluation"
run_command "python3 rook_bb-cio.py -m $model_hf_path -g" "$temp_log2" "ROOK BB-CIO Evaluation"
run_command "python3 rook_selfplay.py -m $model_hf_path" "$temp_log3" "ROOK Self-play Evaluation"
run_command "python3 rook_vs_stockfish.py -m $model_hf_path -g -p $stockfish_path" "$temp_log4" "ROOK vs Stockfish Evaluation"

# Wait for all background processes to finish
wait

# Function to append a log with a header to the aggregate log
append_log() {
    local header="$1"
    local log_file="$2"
    {
        echo "===== $header ====="
        cat "$log_file"
        echo ""  # Add a blank line for separation
    } >> "$aggregate_log"
}

# Clear the aggregate log if it exists
> "$aggregate_log"

# Append each log to the aggregate log with headers
append_log "ROOK Accuracy Evaluation (rook_accuracy.py -m $model_hf_path -d $data_path -g)" "$temp_log1"
append_log "ROOK BB-CIO Evaluation (rook_bb-cio.py -m $model_hf_path -b 1 -g -n 100)" "$temp_log2"
append_log "ROOK Self-play Evaluation (rook_selfplay.py -m $model_hf_path -n $num_games -g)" "$temp_log3"
append_log "ROOK vs Stockfish Evaluation (rook_vs_stockfish.py -m $model_hf_path -n $num_games -g -p $stockfish_path -s 10 -l 0.1 -t -1)" "$temp_log4"

# Remove temporary log files
rm "$temp_log1" "$temp_log2" "$temp_log3" "$temp_log4"

echo "All commands have completed. Aggregate output is in $aggregate_log"
