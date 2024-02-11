from model_config_xz import *
from stt_class_xz import STT
from .utils import *

from logger_config import setup_file_logging
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Insert the audio and text file to be processed.")
    parser.add_argument('-a', '--audio-path', required=True, help='Path to audio files directory.')
    parser.add_argument('-t', '--text-path', required=False, help='Path to text metadata file.')
    args = parser.parse_args()

    language_code = 'es'
    stt = STT(language_code)
    audio_path = Path(args.audio_path)
    text_path = args.text_path if args.text_path else audio_path
    database = create_dir(audio_path)
    
    logger = setup_file_logging(f'{database}/logs/{database}_{language_code}_model.log')

    file_pairs, total_audios = flac_txt_files(audio_path)

    header_info(stt, audio_path, total_audios, 0, logger) 

    results_df = process_audios(stt, audio_path, 0, logger)  # Total words will be counted in the loop

    total_words = results_df['words'].sum()
    calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database, logger)

if __name__ == "__main__":
    main()
