from num2words import num2words
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import re

#################
# PREPROCESSING #
#################
def db_name(path):
    return Path(path).parts[3]

def sub_db(path):
    return Path(path).parts[5]

def section_name(path):
    return Path(path).parts[6]

def create_dir(path):
    database = db_name(path)
    sub_database = sub_db(path)
    section = section_name(path)

    logs_path = f"{database}/logs/{sub_database}"
    results_path = f"{database}/results/{sub_database}"

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
        print(f"Created logs directory: {logs_path}")
    else:
        print(f"Logs directory already exists: {logs_path}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"Created results directory: {results_path}")
    else:
        print(f"Results directory already exists: {results_path}")

    return database, sub_database,section

def replace_numbers_in_text(text):
    for num in re.findall(r'\b\d+\b', text):
        text = text.replace(num, num2words(num, lang='es'), 1)
    return text

def load_data(stt, path):
    wav_files = sorted([file for file in os.listdir(path) if file.endswith('.wav') and 'k_' in file])
    txt_file = next((file for file in os.listdir(path) if file.endswith('.txt')), None)
    if txt_file is None:
        raise ValueError("No transcription txt file found in the directory.")
    
    with open(os.path.join(path, txt_file), 'r') as file:
        lines = file.readlines()
        transcripts = [re.sub(r' \d+(\.\d+)? \d+(\.\d+)?$', '', line).strip() for line in lines]
        transcripts = [replace_numbers_in_text(transcript) for transcript in transcripts]

    if len(transcripts) != len(wav_files):
        raise ValueError("The number of lines in the txt file doesn't match the number of wav files.")

    validation_df = pd.DataFrame({
        'wav_filename': wav_files,
        'transcript': transcripts
    })

    total_words = validation_df['transcript'].apply(lambda x: len(stt.transformation(x).split())).sum()
    return validation_df, total_words


####################
# PROCESSING AUDIO #
####################
def transcribe_audio(stt, audio_path, reference, logger):
    if reference is None or reference.strip() == "":
        logger.info(f"Reference transcription missing or empty for {audio_path}. Skipping.")
        return None
    try:
        hypothesis = stt.run(audio_path)
        if hypothesis is None or hypothesis.strip() == "":
            logger.info(f"Hypothesis missing or empty for {audio_path}. Skipping.")
            return None
    except FileNotFoundError:
        logger.info(f"File {audio_path} does not exist. Skipping.")
        return None
    except OSError as e:
        logger.error(f"OS error occurred when processing file {audio_path}: {e}")
        return None

    reference_transformed = stt.transformation(reference)
    hypothesis_transformed = stt.transformation(hypothesis)

    wer = stt.compute_wer(reference_transformed, hypothesis_transformed)
    word_count = stt.compute_word_count(reference_transformed)
    error_count = stt.compute_error_count(wer, word_count)
    
    return wer, word_count, reference_transformed, hypothesis_transformed, error_count


def process_audios(stt, validation_df, total_audios, path, logger):
    results_df = pd.DataFrame(columns=['audio_file', 'reference', 'hypothesis', 'wer', 'words', 'errors'])

    for idx, row in tqdm(validation_df.iterrows(), total=total_audios, desc="Processing audios"):
    # for idx, row in tqdm(validation_df.head(1).iterrows(), total=total_audios, desc="Processing audios"):
        audio_file = row['wav_filename']
        reference = row['transcript']
        audio_path = path / audio_file

        result = transcribe_audio(stt, audio_path, reference, logger)
        if result is not None:
            wer, word_count, reference_transformed, hypothesis_transformed, error_count = result
            results_df.loc[idx] = [audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count]
            processing_info(idx+1, total_audios, audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count, logger)
    return results_df

def calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database, sub_database, section, logger):
    total_errors = results_df['errors'].sum()
    wwer = total_errors / total_words
    mean_wer = results_df['wer'].mean()

    wwer_info(total_audios, total_words, total_errors, wwer, mean_wer, logger)
    save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database, sub_database, section, logger)

##################
# SAVING RESULTS #
##################
def save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database, sub_database, section, logger):
    final_results_df = pd.DataFrame({
        'model': [stt.config['name'].replace(" ", "_")],
        'language': [stt.lang],
        'database': [database],
        'sub_database': [sub_database],
        'section': [section],
        'total_audios': [total_audios],
        'total_words': [total_words],
        'total_errors': [total_errors],
        'wwer': [wwer],
        'mean_wer': [mean_wer]
    })
    try:
        file_name = f"{database}/results/{sub_database}/{stt.config['name'].replace('.', '_')}_{database}_{sub_database}_{section}.csv"
        file_name = file_name.replace(' ', '_')
        with open(file_name, 'w') as file:
            final_results_df.to_csv(file, index=False)
            logger.info(f"Final results saved in {os.path.abspath(file_name)}")
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

######################
# LOGGER INFORMATION #
######################
def wwer_info(total_audios, total_words, total_errors, wwer, mean_wer, logger):
    logger.info("\nWWER INFO")
    logger.info(f'Total audios: \t{total_audios}')
    logger.info(f'Total words: \t{total_words}')
    logger.info(f"Total errors: \t{total_errors}")
    logger.info(f"Weighted WER: \t{wwer}")
    logger.info(f"Mean WER: \t{mean_wer}")

def header_info(stt, path, total_audios, total_words, logger):
    logger.info("\nHEADER INFO")
    logger.info(f"Model:\t\t {stt.config['name']}")
    logger.info(f"Version:\t {stt.config['version']}")
    logger.info(f"Main DB:\t {db_name(path)}")
    logger.info(f"Sub DB:\t {sub_db(path)}")
    logger.info(f"Section:\t {section_name(path)}")    
    logger.info(f'Total audios:\t {total_audios}')
    logger.info(f"Total words:\t {total_words}")

def processing_info(idx, total_audios, audio_file, reference, hypothesis, wer, word_count, error_count, logger):
    logger.info("\nPROCESSING INFO")
    logger.info(f"Processing audio #{idx} of {total_audios}")
    logger.info(f"Audio file: {audio_file}")
    logger.info(f"\tReference: \t\t\t{reference}")
    logger.info(f"\tHypothesis: \t\t{hypothesis}")
    logger.info(f"\tWord Error Rate: \t\t\t{wer}")
    logger.info(f"\tWords in reference: \t\t{word_count}")
    logger.info(f"\tWrong Words in hypothesis: \t{error_count}")