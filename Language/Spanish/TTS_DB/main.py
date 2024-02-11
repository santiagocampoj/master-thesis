from model_config_xz import *
from stt_class_xz import STT
from .utils import *

from logger_config import setup_file_logging
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Insert the audio and text file to be processed.")
    parser.add_argument('-a', '--audio-path', required=True, help='Path to audio files directory.')
    parser.add_argument('-t', '--text-path', required=False, help='Path to text trancription.')
    args = parser.parse_args()

    language_code = 'es'
    stt = STT(language_code)

    audio_path = Path(args.audio_path)
    database, speaker = create_dir(audio_path)

    logger = setup_file_logging(f'{database}/logs/{speaker}/{database}_{speaker}.log')
    
    if args.text_path:
        validation_df, total_words = load_data(stt, args.text_path, args.audio_path)
    else:
        validation_df, total_words = load_data(stt, combined_path=args.audio_path)

    total_audios = len(validation_df)
    header_info(stt, audio_path, total_audios, total_words, logger)
    
    results_df = process_audios(stt, validation_df, total_audios, audio_path, logger)
    calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database, speaker, logger)

if __name__ == "__main__":
    main()
