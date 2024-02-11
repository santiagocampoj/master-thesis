from model_config_xz import *
from stt_class_xz_copy import WhisperSTT
from .utils import *

from logger_config import setup_file_logging
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Insert the audio and text file to be processed.")
    parser.add_argument('-a', '--audio-path', required=True, help='Path to audio files directory.')
    parser.add_argument('-t', '--text-path', required=True, help='Path to text metadata file.')
    args = parser.parse_args()

    # language_code = 'eu'
    # stt = STT(language_code)
    
    language_code = 'whisper'
    whisper_stt = WhisperSTT(STT_MODELS['whisper'])

    path = Path(args.audio_path)
    database = create_dir(path)

    logger = setup_file_logging(f'{database}/logs/{database}_{language_code}_whisper_model.log')

    validation_df, total_words = load_data(whisper_stt, args.text_path)
    total_audios = len(validation_df)
    sub_db_name = short_db_name(path)
    header_info(whisper_stt, path, total_audios, total_words, sub_db_name, logger)

    results_df = process_audios(whisper_stt, validation_df, total_audios, path, logger)
    calculate_wwer(whisper_stt, results_df, total_audios, total_words, path, database, logger)

if __name__ == "__main__":
    main()
