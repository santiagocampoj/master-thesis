from model_config_general import *

from stt import Model
import pydub
import numpy as np
import jiwer

import logging
logger = logging.getLogger("pydub.converter")
logger.setLevel(logging.WARNING)

class STT:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.transformation = jiwer.Compose([
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase(),
        ])
    
    def load_model(self):
        if self.model is None:
            self.model = Model(self.config['acoustic'])
            self.model.enableExternalScorer(self.config['scorer'])

    def transcribe(self, audio_path, start_time, end_time):
        sound = pydub.AudioSegment.from_file(audio_path)
        segment = sound[start_time * 1000:end_time * 1000]
        if segment.frame_rate != self.model.sampleRate():
            segment = segment.set_frame_rate(self.config['desired_sample_rate'])
        audio = np.array(segment.get_array_of_samples())
        return self.model.stt(audio)

    def compute_wer(self, reference, hypothesis):
        reference_transformed = self.transformation(reference)
        hypothesis_transformed = self.transformation(hypothesis)
        return jiwer.wer(reference_transformed, hypothesis_transformed)

    def compute_word_count(self, reference):
        reference_transformed = self.transformation(reference)
        words = len(reference_transformed.split())
        return words

    def compute_error_count(self, wer, word_count):
        return wer * word_count