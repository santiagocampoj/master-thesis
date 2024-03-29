import os
import time
import logging
import pydub
import numpy as np
from scipy.io import wavfile
import jiwer
import tempfile

from stt import Model
from coqui_stt_model_manager.modelmanager import ModelManager
from model_config_xz import *

import argparse
import whisper
# import model_config_xz
from whisper_evaluate import set_lm_options, parse_transcribe_options

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logging.getLogger("pydub.converter").setLevel(logging.WARNING)

INSTALL_DIR = "/home/aholab/santi/Documents/audio_process/Test/Language/models"

def ensure_samplerate(audio_path, desired_sample_rate):
    try:
        sound = pydub.AudioSegment.from_file(audio_path)
    except pydub.exceptions.CouldntDecodeError as error:
        raise ValueError('Could not decode audio file.') from error

    sample_rate = sound.frame_rate
    if sample_rate != desired_sample_rate:
        sound = sound.set_frame_rate(desired_sample_rate)
        # sound.export(audio_path, format='wav') # this is not used
    return sound

def convert_to_mono(sound):
    if sound.channels > 1:
        sound = sound.split_to_mono()[0]
    return sound

def read_wav(audio_path, desired_sample_rate):
    sound = ensure_samplerate(audio_path, desired_sample_rate)
    sound = convert_to_mono(sound)
    audio = np.array(sound.get_array_of_samples())

    # It needs to be in 16-bit precision:
    if audio.dtype == 'int8':
        audio = np.array(audio.astype(float) * (2**8), dtype=np.int16)
    elif audio.dtype == 'int32':
        audio = np.array(audio.astype(float) / (2**16), dtype=np.int16)
    elif audio.dtype == 'float32':
        audio = np.array(audio.astype(float) / (2**16), dtype=np.int16)
    return audio

class STT:
    def __init__(self, lang, scorer=True):
        self.lang = lang
        if self.lang not in STT_MODELS:
            raise ValueError(f'Unknown language: {self.lang}')
        self.config = STT_MODELS[self.lang]
        if not scorer and 'scorer' in self.config:
            del self.config['scorer']
        
        self.transformation = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.Strip(),
        ])
        
        os.makedirs(INSTALL_DIR, exist_ok=True)

        self.model = None
        logging.info('Downloading %s model...', lang)
        self.download()
        logging.info('Model downloaded.')
        logging.info('Loading %s model...', lang)
        self.load()
        logging.info('Model loaded.')

    def download(self):
        self.manager = ModelManager(install_dir=INSTALL_DIR)
        self.manager.download_model(STT_MODELS[self.lang])

        if not self.config['name'] in self.manager.models_dict():
            logging.info('Waiting for %s to download...', self.config['name'])

            while not self.config['name'] in self.manager.models_dict():
                time.sleep(1)

        self.card = self.manager.models_dict()[self.config['name']]

    def scorer(self, scorer_path):
        if scorer_path is None:
            self.model.disableExternalScorer()
        else:
            self.model.enableExternalScorer(scorer_path)

    def load(self):
        if self.model is not None:
            return
        acoustic_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.card.acoustic_path
        ) # join initial path + acoustic model path
        self.model = Model(acoustic_path)
        if 'scorer' in self.config:
            scorer_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.card.scorer_path
            )
            self.scorer(scorer_path)

    def run(self, audio_path, start_time=None, end_time=None):
        if start_time is not None and end_time is not None:
            sound = pydub.AudioSegment.from_file(audio_path)
            segment = sound[start_time * 1000:end_time * 1000]
            if segment.frame_rate != self.model.sampleRate():
                segment = segment.set_frame_rate(self.model.sampleRate())
            audio = np.array(segment.get_array_of_samples())
        else:
            # logging.debug('[STT:%s] Audio path: %s', self.lang, audio_path)
            desired_sample_rate = self.model.sampleRate()
            audio = read_wav(audio_path, desired_sample_rate)

        text = self.model.stt(audio)
        return text

    def compute_wer(self, reference, hypothesis):
        reference_transformed = self.transformation(reference)
        hypothesis_transformed = self.transformation(hypothesis)
        return jiwer.wer(reference_transformed, hypothesis_transformed)

    def compute_word_count(self, reference):
        reference_transformed = self.transformation(reference)
        words = len(reference_transformed.split())
        return words

    def compute_error_count(self, wer, word_count):
        return int(round(wer * word_count))



class WhisperSTT:
    def __init__(self, model_config):
        self.config = model_config
        self.model = self.load_model()
        self.lang = model_config.get('language_code', 'unknown')

        self.transformation = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.Strip(),
        ])

    def load_model(self):
        model_path = self.config['acoustic']
        return whisper.load_model(model_path)

    def run(self, audio_path, start_time=None, end_time=None, lm_alpha=0.33582368603855817, lm_beta=0.6882556478819416):
        args = argparse.Namespace(lm_alpha=lm_alpha, lm_beta=lm_beta)
        set_lm_options(args)
        transcribe_options = parse_transcribe_options(args)

        logging.info(f"using language model: {self.config.get('language_model', 'Not Specified')}")
        logging.info(f"lm alpha: {lm_alpha}")
        logging.info(f"lm beta: {lm_beta}")

        if start_time is not None and end_time is not None:
            sound = pydub.AudioSegment.from_file(audio_path)
            segment = sound[start_time * 1000:end_time * 1000]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                segment.export(temp_file.name, format="wav")
                audio_path_str = str(temp_file.name)

            result = self.model.transcribe(audio_path_str, **transcribe_options)
            os.remove(audio_path_str)
        else:
            audio_path_str = str(audio_path)
            result = self.model.transcribe(audio_path_str, **transcribe_options)

        return result['text']

    def compute_wer(self, reference, hypothesis):
        transformation = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase(),
            jiwer.Strip(),
        ])
        reference_transformed = transformation(reference)
        hypothesis_transformed = transformation(hypothesis)
        return jiwer.wer(reference_transformed, hypothesis_transformed)

    def compute_word_count(self, text):
        transformation = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase(),
            jiwer.Strip(),
        ])
        text_transformed = transformation(text)
        words = len(text_transformed.split())
        return words

    def compute_error_count(self, wer, word_count):
        return int(round(wer * word_count))