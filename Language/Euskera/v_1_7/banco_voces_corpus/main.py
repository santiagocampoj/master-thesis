from model_config_xz import *
from stt_class_xz import STT
from .utils import *

from logger_config import setup_file_logging
from pathlib import Path
import argparse

logger = logging.getLogger("pydub.converter")
logger.setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="Insert the path for audio and text files to be processed.")
    parser.add_argument('-d', '--directory', required=True, help='Directory to audio and txt files.')
    args = parser.parse_args()

    language_code = 'eu'
    stt = STT(language_code)

    audio_path = Path(args.directory)
    if audio_path.suffix != ".json":
        raise ValueError(f"Unsupported file format: {audio_path.suffix}. Please provide a path to a .json file.")

    database = create_dir(audio_path)

    logger = setup_file_logging(f'{database}/logs/{database}_{language_code}_model.log')
    
    validation_df, total_words = load_data(stt, audio_path)
    total_audios = len(validation_df)
    header_info(stt, audio_path, total_audios, total_words, logger)
    
    results_df = process_audios(stt, validation_df, total_audios, audio_path, logger)
    calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database, logger)

if __name__ == "__main__":
    main()
