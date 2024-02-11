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

def speaker_name(path):
    return Path(path).parts[4]

def create_dir(path):
    database = db_name(path)
    speaker = speaker_name(path)
        
    if not os.path.exists(f'{database}/logs/{speaker}/'):
        os.makedirs(f'{database}/logs/{speaker}/')
    if not os.path.exists(f'{database}/results/{speaker}/'):
        os.makedirs(f'{database}/results/{speaker}/')
    return database, speaker

def clean_text(text):
    text = re.sub(r'ã', 'á', text)
    text = re.sub(r'ï', '', text)
    text = re.sub(r'ã©', 'é', text)
    text = re.sub(r'ã³', 'ó', text)
    text = re.sub(r'á³', 'ó', text)
    text = re.sub(r'á±', 'ñ', text)
    text = re.sub(r'ã', 'í', text)
    text = re.sub(r'á­', 'í', text)
    text = re.sub(r'á©', 'e', text)
    return text

def load_data(stt, prompt_path=None, wave_path=None, combined_path=None):
    if combined_path:
        prompt_path = combined_path
        wave_path = combined_path

    wav_files = [file for file in os.listdir(wave_path) if file.endswith('.wav')]
    if not wav_files:
        raise ValueError("No audio file found in the directory.")

    transcripts = []
    for wav_file in wav_files:
        # print("########################################")
        # print(f"\nwav file: {wav_file}")
        txt_filename_initial = wav_file.split(".")[0] + ".txt"
        # print(f"\ntxt_filename initial: {txt_filename_initial}")
        
        txt_filepath = os.path.join(prompt_path, txt_filename_initial)
        # print(f"\ntxt_filepath {txt_filepath}")

        if not os.path.exists(txt_filepath):
            # print("\n////////////////////////////////")
            # print(f"if file doesnt exist: {txt_filepath}")
            
            patterns = [
                (r'(\d{4})_\d{2}', wav_file),
                (r'(.*?)(\d+)$', wav_file.split('.')[0]),
                (r'([A-Z]{3})(\d{3})', wav_file) 
            ]
            
            match_pattern = None
            for pattern, string in patterns:
                match = re.search(pattern, string, re.IGNORECASE)
                if match:
                    match_pattern = match
                    break

            if match_pattern:
                if pattern == patterns[0][0]:
                    txt_filename = f"{match_pattern.group(1)}.txt"
                elif pattern == patterns[1][0]:
                    prefix = match_pattern.group(1)
                    number_part = int(match_pattern.group(2))
                    txt_filename = f"{prefix}{number_part}.txt"
                else:
                    prefix = match_pattern.group(1)
                    number_part = int(match_pattern.group(2))
                    txt_filename = f"{prefix}{number_part}.txt"
                
                txt_filepath = os.path.join(prompt_path, txt_filename)
                
                # print(f"After match pattern txt_filename: {txt_filename}")
                # print(f"After match pattern txt_filepath: {txt_filepath}\n")

            if not os.path.exists(txt_filepath):
                raise ValueError(f"Transcription file {txt_filepath} does not exist.")

        with open(txt_filepath, 'r', encoding='ISO-8859-15') as txt_file:
            transcript = txt_file.readline().strip()
            transcripts.append(transcript)

    if len(wav_files) != len(transcripts):
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
    if not reference or reference.strip() == "":
        logger.info(f"Reference transcription missing or empty for {audio_path}. Skipping.")
        return None
    try:
        hypothesis = stt.run(audio_path)
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
    n_audios = 1
    for idx, row in tqdm(validation_df.iterrows(), total=total_audios, desc="Processing audios"):
    # for idx, row in tqdm(validation_df.head(n_audios).iterrows(), total=n_audios, desc="Processing audios"):
        audio_file = row['wav_filename']
        reference = row['transcript']
        audio_path = path / audio_file

        result = transcribe_audio(stt, audio_path, reference, logger)
        if result is not None:
            wer, word_count, reference_transformed, hypothesis_transformed, error_count = result
            results_df.loc[idx] = [audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count]
            processing_info(idx+1, total_audios, audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count, logger)
    return results_df

def calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database, speaker, logger):
    total_errors = results_df['errors'].sum()
    wwer = total_errors / total_words
    mean_wer = results_df['wer'].mean()

    wwer_info(total_audios, total_words, total_errors, wwer, mean_wer, logger)
    save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database, speaker, logger)


##################
# SAVING RESULTS #
##################
def save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database, speaker, logger):
    final_results_df = pd.DataFrame({
        'model': [stt.config['name']],
        'language': [stt.lang],
        'database': [database],
        'speaker': [speaker],
        'total_audios': [total_audios],
        'total_words': [total_words],
        'total_errors': [total_errors],
        'wwer': [wwer],
        'wer': [mean_wer]
    })
    try:
        file_name = f"{database}/results/{speaker}/{stt.config['name'].replace('.', '_')}_{database}_{speaker}.csv"
        file_name = file_name.replace(' ', '_')
        with open(file_name, 'w') as file:
            final_results_df.to_csv(file, index=False)
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
    logger.info(f"WER: \t{mean_wer}")

def header_info(stt, path, total_audios, total_words, logger):
    logger.info("\nHEADER INFO")
    logger.info(f"Model:\t\t {stt.config['name']}")
    logger.info(f"Version:\t {stt.config['version']}")
    logger.info(f"Main DB:\t {db_name(path)}")
    logger.info(f"Speaker:\t {speaker_name(path)}")
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