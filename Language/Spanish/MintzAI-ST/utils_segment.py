import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence
from stt_class_xz import *
#################
# PREPROCESSING #
#################
def db_name(path):
    return Path(path).parts[3].split('.')[0]

def create_dir(path, logger):
    database = db_name(path)
    logs_path = f'{database}/logs/'
    results_path = f'{database}/results/'

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
        logger.info(f"Created logs directory: {logs_path}")
    else:
        logger.info(f"Logs directory already exists: {logs_path}")

    if not os.path.exists(results_path):
        os.makedirs(results_path)
        logger.info(f"Created results directory: {results_path}")
    else:
        logger.info(f"Results directory already exists: {results_path}")

    return database

def load_data(stt, prompt_path, wave_path, logger):
    wav_files = [file for file in os.listdir(wave_path) if file.endswith('.m4a')]
    if not wav_files:
        logger.error("No audio file found in the directory.")
        raise ValueError("No audio file found in the directory.")

    logger.info(f"Found {len(wav_files)} audio files.")

    transcripts = []
    for wav_file in wav_files:
        txt_filename = wav_file.replace('.m4a', '.es')
        txt_filepath = os.path.join(prompt_path, txt_filename)
        
        if not os.path.exists(txt_filepath):
            logger.error(f"Transcription file {txt_filepath} does not exist.")
            raise ValueError(f"Transcription file {txt_filepath} does not exist.")

        with open(txt_filepath, 'r', encoding='utf-8') as txt_file:
            transcript = txt_file.readline().strip()
            transcripts.append(transcript)
        logger.info(f"Loaded transcript for {wav_file}")

    if len(wav_files) != len(transcripts):
        logger.error("The number of lines in the txt file doesn't match the number of wav files.")
        raise ValueError("The number of lines in the txt file doesn't match the number of wav files.")

    validation_df = pd.DataFrame({
        'wav_filename': wav_files,
        'transcript': transcripts
    })

    total_words = validation_df['transcript'].apply(lambda x: len(stt.transformation(x).split())).sum()
    logger.info(f"Total words in transcripts: {total_words}")

    return validation_df, total_words

####################
# PROCESSING AUDIO #
####################
def segment_audio(audio_path, logger, min_silence_len=500, silence_thresh=-30, keep_silence=200):
    sound = AudioSegment.from_file(audio_path)
    chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    logger.info(f"Total segments created: {len(chunks)}")
    return chunks

def transcribe_chunks(stt, chunks, sample_rate, logger):
    transcriptions = []
    for i, chunk in enumerate(chunks):
        temp_file_path = f"/tmp/chunk_{i}.wav"
        chunk.export(temp_file_path, format="wav")
        logger.info(f"Exporting chunk {i} to {temp_file_path}")

        audio_data = read_wav(temp_file_path, sample_rate)
        transcription = stt.model.stt(audio_data)
        transcriptions.append(transcription)

        os.remove(temp_file_path)
        logger.info(f"Processed and removed temporary file: {temp_file_path}")
    return transcriptions

def transcribe_audio(stt, audio_path, reference, logger):
    try:
        sound = AudioSegment.from_file(audio_path)
        duration_seconds = len(sound) / 1000

        if duration_seconds >= 11:
            logger.info(f"Audio is {duration_seconds} long, which is over 11 seconds, segmenting audio")
            chunks = segment_audio(audio_path, logger)
            transcriptions = transcribe_chunks(stt, chunks, stt.model.sampleRate(), logger=logger)
            hypothesis = " ".join(transcriptions)  # concatenate transcriptions
        else:
            logger.info(f"Audio is {duration_seconds} long, which is less than 11 seconds, processing whole audio")
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

    logger.info(f"Transcription completed for {audio_path}")
    
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

    file_name = f"{database}/results/{stt.config['name'].replace(' ', '_')}_{database}.csv"
    with open(file_name, 'w') as file:
        final_results_df.to_csv(file, index=False)


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
    logger.info(f"\tWrong Words: \t\t\t\t{error_count}")