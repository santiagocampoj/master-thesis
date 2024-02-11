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
        for ses_directory in "$block_directory"/*; do
            if [ -d "$ses_directory" ]; then
                ses_dir_name=$(basename "$ses_directory")
                log_folder="/home/aholab/santi/Documents/audio_process/Language/Euskera/v_1_8/ADITU/nohup_logs/${block_dir_name}"
                log_path="${log_folder}/${ses_dir_name}.log"

                echo "Processing directory: $ses_directory"
                if $use_nohup; then
                    mkdir -p "$log_folder"
                    nohup python3 -m ADITU.main -a "$ses_directory" -t "$ses_directory" > "$log_path" 2>&1 &
                    wait 
                else
                    python3 -m ADITU.main -a "$ses_directory" -t "$ses_directory"
                fi
            fi
        done
    fi
done