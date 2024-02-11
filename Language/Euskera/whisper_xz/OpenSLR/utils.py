import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import codecs


def db_name(path):
    return Path(path).parts[3]

def long_db_name(path):
    return '_'.join(Path(path).parts[3:6])

def short_db_name(path):
    return '_'.join(Path(path).parts[4:7])

def create_dir(path):
    db = db_name(path)
    db_long = long_db_name(path)

    if not os.path.exists(f'{db}/logs/'):
        os.makedirs(f'{db}/logs/')
    if not os.path.exists(f'{db}/results/'):
        os.makedirs(f'{db}/results/')
    return db, db_long

def load_data(stt, text_path):
    txt_files_count = 0
    all_data = []
    for txt_file in os.listdir(text_path):
        txt_files_count = txt_files_count + 1
        with codecs.open(os.path.join(text_path, txt_file), 'r', encoding='utf-8-sig') as file:
            transcription = file.readline()
            audio_path = txt_file.replace("txt", "wav")
            all_data.append({
                    'wav_filename': audio_path,
                    'transcript': transcription 
                })
    
    df = pd.DataFrame(all_data)
    total_words = df['transcript'].apply(lambda x: len(stt.transformation(x).split())).sum()
    return df, total_words

####################
# PROCESSING AUDIO #
####################
def transcribe_audio(stt, audio_path, reference, logger):
    try:
        hypothesis = stt.run(audio_path)
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

    # for idx, row in tqdm(validation_df.head(5).iterrows(), total=5, desc="Processing audios"):
    for idx, row in tqdm(validation_df.iterrows(), total=total_audios, desc="Processing audios"):
        audio_file = row['wav_filename']
        reference = row['transcript']
        audio_path = path / audio_file

        result = transcribe_audio(stt, audio_path, reference, logger)
        if result is not None:
            wer, word_count, reference_transformed, hypothesis_transformed, error_count = result
            results_df.loc[idx] = [audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count]
            processing_info(stt, idx+1, total_audios, audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count, logger)
    
    return results_df

def calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database, logger):
    total_errors = results_df['errors'].sum()
    wwer = total_errors / total_words
    mean_wer = results_df['wer'].mean()

    wwer_info(total_audios, total_words, total_errors, wwer, mean_wer, logger)
    save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database, logger)


##################
# SAVING RESULTS #
##################
def save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database, logger):
    final_results_df = pd.DataFrame({
        'model': [stt.config['name'].replace(" ", "_")],
        'language': [stt.lang],
        'database': [database],
        'total_audios': [total_audios],
        'total_words': [total_words],
        'total_errors': [total_errors],
        'wwer': [wwer],
        'mean_wer': [mean_wer]
    })

    file_name = f"{database}/results/{stt.config['name'].replace(' ', '_')}_{database}.csv"
    with open(file_name, 'w') as file:
        final_results_df.to_csv(file, index=False)


######################
# LOGGER INFORMATION #
######################
def wwer_info(total_audios, total_words, total_errors, wwer, mean_wer, logger):
    logger.info(f"\nWWER INFO")
    logger.info(f'Total audios: \t{total_audios}')
    logger.info(f'Total words: \t{total_words}')
    logger.info(f"Total errors: \t{total_errors}")
    logger.info(f"Weighted WER: \t{wwer}")
    logger.info(f"Mean WER: \t\t{mean_wer}")

def header_info(stt, path, total_audios, total_words, sub_db_name, logger):
    logger.info(f"\nHEADER INFO")
    logger.info(f"Model:\t\t {stt.config['name']}")
    logger.info(f"Main DB:\t {db_name(path)}")
    logger.info(f"Sub DB:\t {sub_db_name}")
    logger.info(f'Total audios:\t {total_audios}')
    logger.info(f"Total words:\t {total_words}")

def processing_info(stt, idx, total_audios, audio_file, reference, hypothesis, wer, word_count, error_count, logger):
    logger.info(f"\nPROCESSING INFO")
    logger.info(f"Processing audio #{idx} of {total_audios}")
    logger.info(f"Audio file: {audio_file}")
    logger.info(f"\tReference: \t\t\t{reference}")
    logger.info(f"\tHypothesis: \t\t{hypothesis}")
    logger.info(f"\tWord Error Rate: \t\t\t{wer}")
    logger.info(f"\tWords in reference: \t\t{word_count}")
    logger.info(f"\tWrong Words in reference: \t{error_count}")