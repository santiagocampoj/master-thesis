from model_config_xz import *
from stt_class_xz_copy import WhisperSTT
from .utils import *

from logger_config import setup_file_logging
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Insert the audio and text file to be processed.")
    parser.add_argument('-a', '--audio-path', required=True, help='Path to audio files directory.')
    parser.add_argument('-t', '--text-path', required=False, help='Path to text trancription.')
    args = parser.parse_args()

    # language_code = 'eu'
    # stt = STT(language_code)
    
    language_code = 'whisper'
    whisper_stt = WhisperSTT(STT_MODELS['whisper'])

    audio_path = Path(args.audio_path)
    database, speaker = create_dir(audio_path)

    logger = setup_file_logging(f'{database}/logs/{speaker}/{database}_{speaker}_whisper.log')
    
    if args.text_path:
        validation_df, total_words = load_data(whisper_stt, args.text_path, args.audio_path)
    else:
        validation_df, total_words = load_data(whisper_stt, combined_path=args.audio_path)

    total_audios = len(validation_df)
    header_info(whisper_stt, audio_path, total_audios, total_words, logger)
    
    results_df = process_audios(whisper_stt, validation_df, total_audios, audio_path, logger)
    calculate_wwer(whisper_stt, results_df, total_audios, total_words, audio_path, database, speaker, logger)

if __name__ == "__main__":
    main()
