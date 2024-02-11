import pandas as pd
import os
from pathlib import Path
import codecs
from tqdm import tqdm
import time
#################
# PREPROCESSING #
#################
def db_name(path):
    return Path(path).parts[3]

def block_name(path):
    return '_'.join(Path(path).parts[5:6])

def ses_name(path):
    return '_'.join(Path(path).parts[6:7])

def create_dir(path):
    db = db_name(path)
    block = block_name(path)
    ses = ses_name(path)

    if not os.path.exists(f'{db}/logs/{block}'):
        os.makedirs(f'{db}/logs/{block}')
    if not os.path.exists(f'{db}/results/{block}'):
        os.makedirs(f'{db}/results/{block}')
    return db, block, ses

def extract_transcriptions(spl_content):
    validation_section = spl_content.split("[Validation states]")[1].strip().split("\n")
    # print(f"\nPrinting validation section: \n{validation_section}")
    transcriptions = []
    for line in validation_section:
        parts = line.split(">-<")
        # print(f"\n{parts}")
        if len(parts) >= 16:  # ensure it's a valid line with enough data
            # print(f"Lenght parts: {len(parts)}")
            # time.sleep(5)
            # for idx, wav in enumerate(parts):
            #     print(idx, wav)
            wav_file = parts[9].strip().replace(".WAV", ".wav")
            # print(f"\nwave file: {wav_file}")

            transcription_parts = parts[0].strip().split('=')
            if len(transcription_parts) > 1:
                transcription = transcription_parts[1].strip()
                # print(f"transcriptions: {transcription}")
            else:
                transcription = transcription_parts[0].strip()  # Fallback in case there's no '='
                # print(f"transcriptions: {transcription}")
            transcriptions.append((wav_file, transcription))
    return transcriptions

def load_data(stt, text_path):
    all_data = []
    for spl_file in os.listdir(text_path):
        if spl_file.endswith(".spl"):
            try:
                with codecs.open(os.path.join(text_path, spl_file), 'r', encoding='utf-8-sig') as file:
                    content = file.read()
            except UnicodeDecodeError:
                with codecs.open(os.path.join(text_path, spl_file), 'r', encoding='ISO-8859-1') as file:
                    content = file.read()

            transcriptions = extract_transcriptions(content)
            # print(f"Transcriptions from {spl_file}: {transcriptions}")
            for wav_filename, transcript in transcriptions:
                all_data.append({
                    'wav_filename': wav_filename,
                    'transcript': transcript 
                })

    # print(f"All data: {all_data[:5]}") 
    df = pd.DataFrame(all_data)
    if 'transcript' not in df.columns:
        print("Error: 'transcript' column not found in DataFrame")
    total_words = df['transcript'].apply(lambda x: len(stt.transformation(x).split())).sum()
    return df, total_words

####################
# PROCESSING AUDIO #
####################
def transcribe_audio(stt, audio_path, reference, logger, start_time=None, end_time=None):
    if not reference or reference.strip() == "":
        logger.info(f"Reference transcription missing or empty for {audio_path}. Skipping.")
        return None
    try:
        hypothesis = stt.run(audio_path, start_time=start_time, end_time=end_time)
        if not hypothesis or hypothesis.strip() == "":
            logger.info(f"Hypothesis missing or empty for {audio_path}. Skipping.")
            return None

        reference_transformed = stt.transformation(reference)
        hypothesis_transformed = stt.transformation(hypothesis)

        if not reference_transformed.strip() or not hypothesis_transformed.strip():
            logger.info(f"Empty transformed reference or hypothesis for {audio_path}. Skipping WER calculation.")
            return None

        wer = stt.compute_wer(reference_transformed, hypothesis_transformed)
        word_count = stt.compute_word_count(reference_transformed)
        error_count = stt.compute_error_count(wer, word_count)
    
    except FileNotFoundError:
        logger.info(f"File {audio_path} does not exist. Skipping.")
        return None
    except OSError as e:
        logger.error(f"OS error occurred when processing file {audio_path}: {e}")
        return None

    return wer, word_count, reference_transformed, hypothesis_transformed, error_count

def process_audios(stt, validation_df, total_audios, path, logger):
    results_df = pd.DataFrame(columns=['audio_file', 'reference', 'hypothesis', 'wer', 'words', 'errors'])

    # for idx, row in tqdm(validation_df.head(15).iterrows(), total=15, desc="Processing audios"):
    for idx, row in tqdm(validation_df.iterrows(), total=total_audios, desc="Processing audios"):
        # if idx > 8:
        audio_file = row['wav_filename']
        reference = row['transcript']
        audio_path = path / audio_file

        result = transcribe_audio(stt, audio_path, reference, logger)
        if result is not None:
            wer, word_count, reference_transformed, hypothesis_transformed, error_count = result
            results_df.loc[idx] = [audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count]
            processing_info(stt, idx+1, total_audios, audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count, logger)
    return results_df

def calculate_wwer(stt, results_df, total_audios, total_words, database, block, ses, logger):
    total_errors = results_df['errors'].sum()
    wwer = total_errors / total_words
    mean_wer = results_df['wer'].mean()

    wwer_info(total_audios, total_words, total_errors, wwer, mean_wer, logger)
    save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, database, block, ses, logger)

##################
# SAVING RESULTS #
##################
def save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, database, block, ses, logger):
    final_results_df = pd.DataFrame({
        'model': [stt.config['name'].replace(" ", "_")],
        'language': [stt.lang],
        'database': [database],
        'block': [block],
        'ses': [ses],
        'total_audios': [total_audios],
        'total_words': [total_words],
        'total_errors': [total_errors],
        'wwer': [wwer],
        'mean_wer': [mean_wer]
    })
    try:
        file_name = f"{database}/results/{block}/{stt.config['name'].replace(' ', '_')}_{database}_{block}_{ses}.csv"
        with open(file_name, 'w') as file:
            final_results_df.to_csv(file, index=False)
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

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

def header_info(stt, path, total_audios, total_words, block, ses, logger):
    logger.info(f"\nHEADER INFO")
    logger.info(f"Model:\t\t {stt.config['name']}")
    logger.info(f"Main DB:\t {db_name(path)}")
    logger.info(f"Block:\t\t {block}")
    logger.info(f"Ses:\t\t {ses}")
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