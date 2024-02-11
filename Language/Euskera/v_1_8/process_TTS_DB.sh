#!/bin/bash

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

if [ -z "$parent_directory" ]; then
    echo "Please provide the parent directory using the -p option."
    exit 1
fi

for block_directory in "$parent_directory"/*; do
    if [ -d "$block_directory" ]; then
        block_dir_name=$(basename "$block_directory")

        # Check if the directory should be processed with a single path or separate wav/txt paths
        if [ "$block_dir_name" == "urkullu_eu" ]; then
            command="python3 -m TTS_DB.main -a $block_directory"
        else
            command="python3 -m TTS_DB.main -a $block_directory/wav/ -t $block_directory/txt/"
        fi

        log_folder="/home/aholab/santi/Documents/audio_process/Language/Euskera/v_1_8/TTS_DB/nohup_logs/${block_dir_name}"
        log_path="${log_folder}/${block_dir_name}.log"

        echo "Running command: $command"
        if $use_nohup; then
            mkdir -p "$log_folder"
            nohup $command > "$log_path" 2>&1 &
            wait
        else
            $command
        fi
    fi
done