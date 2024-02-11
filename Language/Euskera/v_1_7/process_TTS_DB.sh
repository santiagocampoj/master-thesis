#!/bin/bash

# Activate the 'stt' Conda environment if not already active
if [[ $CONDA_DEFAULT_ENV != "stt" ]]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate stt
    echo "stt conda environment is setup"
fi

use_nohup=false
parent_directory=""

while getopts ":p:n" opt; do
  case $opt in
    p)
      parent_directory="$OPTARG"
      ;;
    n)
      use_nohup=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check if parent directory is provided
if [ -z "$parent_directory" ]; then
    echo "Please provide the parent directory using the -p option."
    exit 1
fi

# Iterate through each speaker directory
for speaker_dir in "$parent_directory"/*; do
    if [[ -d "$speaker_dir" && "$speaker_dir" == *"eu" ]]; then
        wav_path="$speaker_dir/wav"
        txt_path="$speaker_dir/txt"

        # Check if both wav and txt subdirectories exist
        if [ -d "$wav_path" ] && [ -d "$txt_path" ]; then
            log_folder="/home/aholab/santi/Documents/audio_process/Language/Spanish/nohup_logs/$(basename "$speaker_dir")"
            log_path="${log_folder}/processing.log"

            echo "Processing Spanish directory: $speaker_dir"
            mkdir -p "$log_folder" # Create log directory if it doesn't exist

            if $use_nohup; then
                nohup python3 -m TTS_DB.main -a "$wav_path" -t "$txt_path" > "$log_path" 2>&1 &
                wait # Wait for the background process to finish before continuing
            else
                python3 -m TTS_DB.main -a "$wav_path" -t "$txt_path"
            fi
        else
            echo "wav or txt directory missing in $speaker_dir"
        fi
    fi
done
