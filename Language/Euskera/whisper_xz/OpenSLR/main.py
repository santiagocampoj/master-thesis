from model_config_xz import *
from stt_class_xz_copy import WhisperSTT
from .utils import *

from logger_config import setup_file_logging
from pathlib import Path
import argparse

def main():
    # set parser
    parser = argparse.ArgumentParser(description="Insert the audio and text file to be processed.")
    parser.add_argument('-a', '--audio-path', required=True, help='Path to audio files directory.')
    parser.add_argument('-t', '--text-path', required=True, help='Path to text metadata file.')
    args = parser.parse_args()

    # language_code = 'eu'
    # stt = STT(language_code)
    
    language_code = 'whisper'
    whisper_stt = WhisperSTT(STT_MODELS['whisper'])

    audio_path = Path(args.audio_path)
    text_path = Path(args.text_path)

    database, database_long = create_dir(audio_path)

    logger = setup_file_logging(f'{database}/logs/{database_long}_{language_code}_model.log')

    validation_df, total_words = load_data(whisper_stt, text_path)
    total_audios = len(validation_df)
    sub_db_name = short_db_name(audio_path)
    header_info(whisper_stt, audio_path, total_audios, total_words, sub_db_name, logger)

    results_df = process_audios(whisper_stt, validation_df, total_audios, audio_path, logger)
    calculate_wwer(whisper_stt, results_df, total_audios, total_words, audio_path, database, logger)

if __name__ == "__main__":
    main()