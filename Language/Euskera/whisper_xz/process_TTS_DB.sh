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

# if parent directory is provided
if [ -z "$parent_directory" ]; then
    echo "Please provide the parent directory using the -p option."
    exit 1
fi

# each speaker directory
for speaker_dir in "$parent_directory"/*; do
    
    log_folder="/home/aholab/santi/Documents/audio_process/Language/Euskera/whisper_xz/TTS_DB/nohup_logs/$(basename "$speaker_dir")"
    log_path="$log_folder.log"

    if [[ -d "$speaker_dir" && "$speaker_dir" == *"urkullu_eu"* ]]; then
        echo "Launching process for urkullu_eu"
        mkdir -p "$log_folder"

        if $use_nohup; then
          nohup python3 -m TTS_DB.main -a /mnt/corpus/TTS_DB/urkullu_eu/ > $log_path 2>&1 &
        else
          python3 -m TTS_DB.main -a /mnt/corpus/TTS_DB/urkullu_eu/
        fi

    elif [[ -d "$speaker_dir" && "$speaker_dir" == *"eu"* ]]; then
        wav_path="$speaker_dir/wav"
        txt_path="$speaker_dir/txt"

        # if both wav and txt subdirectories exist
        if [ -d "$wav_path" ] && [ -d "$txt_path" ]; then
            echo "Processing directory: $speaker_dir"
            mkdir -p "$log_folder" # create log directory if it doesn't exist

            if $use_nohup; then
                nohup python3 -m TTS_DB.main -a "$wav_path" -t "$txt_path" > "$log_path" 2>&1 &
                wait # wait for the background process to finish before continuing
            else
                python3 -m TTS_DB.main -a "$wav_path" -t "$txt_path"
            fi
        else
            echo "wav or txt directory missing in $speaker_dir"
        fi
    fi
done




# # each speaker directory
# for speaker_dir in "$parent_directory"/*; do
    
#     log_folder="/home/aholab/santi/Documents/audio_process/Language/Euskera/whisper_xz/TTS_DB/nohup_logs/$(basename "$speaker_dir")"
#     log_path="$log_folder.log"
#     mkdir -p "$log_folder" # Ensure log directory exists before processing

#     wav_path="$speaker_dir/wav"
#     txt_path="$speaker_dir/txt"

#     if [ -d "$wav_path" ] && [ -d "$txt_path" ]; then
#         echo "Processing directory: $speaker_dir"

#         if $use_nohup; then
#             nohup python3 -m TTS_DB.main -a "$wav_path" -t "$txt_path" > "$log_path" 2>&1 &
#         else
#             python3 -m TTS_DB.main -a "$wav_path" -t "$txt_path"
#         fi
#     else
#         echo "wav or txt directory missing in $speaker_dir, treating as special case like urkullu_eu"
#         if [[ "$speaker_dir" == *"eu"* ]]; then
#             if $use_nohup; then
#                 nohup python3 -m TTS_DB.main -a "$speaker_dir" > "$log_path" 2>&1 &
#             else
#                 python3 -m TTS_DB.main -a "$speaker_dir"
#             fi
#         fi
#     fi
# done

