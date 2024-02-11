#!/bin/bash

if [[ $CONDA_DEFAULT_ENV != "whisper" ]]; then
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

for sub_directory in "$parent_directory"/*; do
    if [ -d "$sub_directory" ]; then
        sub_dir_name=$(basename "$sub_directory")
        for nested_sub_directory in "$sub_directory"/*; do
            if [ -d "$nested_sub_directory" ]; then
                nested_sub_dir_name=$(basename "$nested_sub_directory")
                log_folder="/home/aholab/santi/Documents/audio_process/Language/Euskera/whisper_xz/Parlamento_EJ/nohup_logs/${sub_dir_name}"
                log_path="${log_folder}/${nested_sub_dir_name}.log"

                echo "Processing directory: $nested_sub_directory"
                if $use_nohup; then
                    mkdir -p "$log_folder"
                    nohup python3 -m Parlamento_EJ.main -d "$nested_sub_directory" > "$log_path" 2>&1 &
                    wait
                else
                    python3 -m Parlamento_EJ.main -d "$nested_sub_directory"
                fi
            fi
        done
    fi
done

