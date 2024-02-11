import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re

#################
# PREPROCESSING #
#################
def db_name(path):
    return Path(path).parts[3].split('.')[0]

def create_dir(path):
    database = db_name(path)
    if not os.path.exists(f'{database}/logs/'):
        os.makedirs(f'{database}/logs/')
    if not os.path.exists(f'{database}/results/'):
        os.makedirs(f'{database}/results/')
    return database

def parse_xml(xml_file):
    """Parse the XML file and return the transcription data."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_data = []

    for unit in root.findall(".//UNIT"):
        start_time = float(unit.attrib["startTime"])
        end_time = float(unit.attrib["endTime"])
        transcription = unit.text if unit.text else ""  # noneyype
        transcription = transcription.strip()
        all_data.append({
            'transcript': transcription,
            'start_time': start_time,
            'end_time': end_time
        })
    return all_data

def clean_transcriptions(transcription):
    transcription = transcription.replace("¤", "").replace("=", "").replace("xxx", "").replace("hhh", "")
    alt_pattern = re.compile(r'{%.*?%}')
    transcription = re.sub(alt_pattern, '', transcription)
    comment_pattern = re.compile(r'\{.*?\}')
    transcription = re.sub(comment_pattern, '', transcription)
    transcription = transcription.replace(" / ", " ").replace(" // ", " ").replace("[/]", "")
    transcription = transcription.replace(">", "").replace("eh", "").replace("mm", "").replace("&eh", "").replace("ah e a", "")
    transcription = " ".join(transcription.split()).strip()
    transcription = transcription.lstrip()
    return transcription

def load_data(stt, prompt_path, wave_path):
    all_data = []
    
    for xml_file in os.listdir(prompt_path):
        if xml_file.endswith('.xml'):
            print(f"Processing ---> {xml_file}")
            parsed_data = parse_xml(os.path.join(prompt_path, xml_file))
            for entry in parsed_data:
                entry['transcript'] = clean_transcriptions(entry['transcript'])
                full_audio_path = os.path.join(wave_path, xml_file.replace('.xml', '.wav'))
                if os.path.exists(full_audio_path):
                    entry['wav_filename'] = full_audio_path
                    all_data.append(entry)

    df = pd.DataFrame(all_data)
    total_words = df['transcript'].apply(lambda x: len(stt.transformation(x).split())).sum()
    return df, int(total_words)

####################
# PROCESSING AUDIO #
####################
def transcribe_audio(stt, audio_path, reference, start_time, end_time, logger):
    if reference is None or reference.strip() == "":
        logger.info(f"Reference transcription missing or empty for {audio_path}. Skipping.")
        return None
    try:
        hypothesis = stt.run(audio_path, start_time=start_time, end_time=end_time)
        if hypothesis is None or hypothesis.strip() == "":
            logger.info(f"Hypothesis missing or empty for {audio_path}. Skipping.")
            return None
    except FileNotFoundError:
        logger.info(f"File {audio_path} does not exist. Skipping.")
        return None
    except OSError as e:
        logger.error(f"OS error occurred when processing file {audio_path}: {e}")
        return None

    try:
        reference_transformed = stt.transformation(reference)
        hypothesis_transformed = stt.transformation(hypothesis)

        wer = stt.compute_wer(reference_transformed, hypothesis_transformed)
        word_count = stt.compute_word_count(reference_transformed)
        error_count = stt.compute_error_count(wer, word_count)
    
        return wer, word_count, reference_transformed, hypothesis_transformed, error_count
    except ValueError as e:
        logger.error(f"Error computing WER for file {audio_path}: {e}")
        return None

def process_audios(stt, validation_df, total_audios, path, logger):
    results_df = pd.DataFrame(columns=['audio_file', 'reference', 'hypothesis', 'wer', 'words', 'errors'])

    for idx, row in tqdm(validation_df.iterrows(), total=total_audios, desc="Processing audios"):
    # for idx, row in tqdm(validation_df.head(1).iterrows(), total=total_audios, desc="Processing audios"):
        # if idx < 520:
        #     continue
        audio_file = row['wav_filename']
        reference = row['transcript']
        audio_path = path / audio_file

        result = transcribe_audio(stt, audio_path, reference, row['start_time'], row['end_time'], logger)
        if result is not None:
            wer, word_count, reference_transformed, hypothesis_transformed, error_count = result
            results_df.loc[idx] = [audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count]
            processing_info(idx+1, total_audios, audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count, logger)
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
    try:
        file_name = f"{database}/results/{stt.config['name'].replace('.', '_')}_{database}.csv"
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
    logger.info(f"\tWrong Words in hypothesis: \t\t\t{error_count}")