from model_config_xz import *
from stt_class_xz import STT
from .utils import *

from logger_config import setup_file_logging
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Insert the audio and text file to be processed.")
    parser.add_argument('-d', '--db-directory', required=True, help='Path to database files directory.')
    args = parser.parse_args()

    language_code = 'eu'
    stt = STT(language_code)

    path = Path(args.db_directory)
    database, sub_database, section = create_dir(path)
    
    # Setting up the logger using the function from logger_config.py
    logger = setup_file_logging(f'{database}/logs/{sub_database}/{section}.log')
    
    validation_df, total_words = load_data(stt, path)
    total_audios = len(validation_df)
    header_info(stt, path, total_audios, total_words, logger)
    
    results_df = process_audios(stt, validation_df, total_audios, path, logger)
    calculate_wwer(stt, results_df, total_audios, total_words, path, database, sub_database, section, logger)

if __name__ == "__main__":
    main()