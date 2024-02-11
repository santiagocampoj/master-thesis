#!/usr/bin/env python
"""Simple script to evaluate Whisper in external audio datasets, not in HF.

The dataset is expected to have the transcriptions in `*.txt` files with the
same name as the audio files.

Example
-------
>>> ./external_dataset_evaluate.py \
    --beam_size 5 \
    --lm_path 5gram-eu.bin \
    --lm_alpha 0.33582368603855817 \
    --lm_beta 0.6882556478819416 \
    --normalize-text \
    --normalize-diacritics \
    ./zuazo-whisper-tiny-eu.pt \
    ./banco_voces_corpus
"""

import os
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import jiwer
import whisper
import numpy as np
from tqdm import tqdm
from unidecode import unidecode
from torch.utils.data import Dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import whisper_decoder_with_lm  # pylint: disable=unused-import # noqa: E501,F401
from whisper_evaluate import parse_transcribe_options, set_lm_options


def normalize_diacritics(text):
    """Remove the diacritics from text.

    Parameters
    ----------
    text : str
        The input text to normalize.

    Returns
    -------
    str:
        The normalized text.
    """
    text = unidecode(text)
    return text


def normalize_text(text, normalizer=None):
    """Normalize text and converts it to lowercase.

    Parameters
    ----------
    text : str
        The input text to normalize.
    normalizer : callable, optional
        External additional function to use to normalize at first step.

    Returns
    -------
    str:
        The normalized text.
    """
    if normalizer is not None:
        text = normalizer(text)
    text.strip().lower()
    return text


class ASRDataset(Dataset):
    """Simple ASR dataset class to work with generic audio datasets.

    This is designed to work with datasets with wav files including their
    text in txt files with the same name.
    """

    def __init__(self, data_dir):
        """ASRDataset constructor.

        Parameters
        ----------
        data_dir : str
            The directory containing the dataset.
        normalize_text : bool
            Whether to normalize text (lowercase and remove diacritics).
            Default is True.
        """
        self.data_dir = data_dir
        self.data = self.load_data()

    def load_data(self):
        """Load data from the dataset directory, pairing audio paths with
        transcriptions.

        Returns
        -------
        list
            A list of tuples, each containing the path to an audio file and its
            transcription.
        """
        data = []
        for root, _, files in os.walk(self.data_dir):
            for filename in files:
                if not filename.endswith(".wav"):
                    continue
                audio_path = os.path.join(root, filename)
                txt_path = os.path.join(root, Path(audio_path).stem + ".txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as handle:
                        text = handle.read().strip()
                    data.append((audio_path, text))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        return audio_path, text


def parse_args():
    """Parses command line arguments.

    Returns
    -------
    namespace
        The namespace populated with the argument values.
    """
    parser = argparse.ArgumentParser(
        description="Speech-to-Text model evaluation script."
    )
    parser.add_argument(
        "model", help="File of the model to use in OpenAI format."
    )  # noqa: E501
    parser.add_argument("dataset", help="Path of the dataset.")
    parser.add_argument(
        "--normalize-audio",
        "-na",
        action="store_true",
        help="Whether normalize the audio file (not recommended).",
    )
    parser.add_argument(
        "--normalize-text",
        "-nt",
        action="store_true",
        help="Whether normalize the text (recommended).",
    )
    parser.add_argument(
        "--normalize-diacritics",
        "-nd",
        action="store_true",
        help="Whether normalize the text diacritics.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Temperature is a form of controlled randomness. "
            "Defaults to 0, which means disabled. The logits will be divided "
            "by this number. "
            "`> 1.0` leads to a more random sampling behaviour. "
            "`< 1.0` makes model more confident in its predictions and "
            "reducing randomness."
        ),
    )
    parser.add_argument(
        "--best_of",
        type=int,
        default=None,
        help="Number of independent sample trajectories (Beam Search).",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=None,
        help="Number of beams in beam search, enables Beam Search.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Patience in beam search.",
    )
    parser.add_argument(
        "--lm_path",
        type=str,
        default=None,
        help="A KenLM n-gram language model path.",
    )
    parser.add_argument(
        "--lm_alpha",
        type=float,
        default=None,
        help="KenLM Language Model weight.",
    )
    parser.add_argument(
        "--lm_beta",
        type=float,
        default=None,
        help="KenLM word insertion weight.",
    )
    parser.add_argument(
        "--lm_eos",
        type=str,
        default=None,
        help="KenLM End-of-String characters.",
    )
    parser.add_argument(
        "--lm_normalize",
        type=bool,
        default=True,
        help="Whether to normalize the text for the KenLM.",
    )
    parser.add_argument(
        "--lm_token_threshold",
        type=int,
        default=None,
        help=(
            "Minimum number of tokens in a sequence required before applying "
            "language model scoring. This prevents premature evaluation on "
            "short sequences."
        ),
    )
    levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    parser.add_argument("--log-level", default="WARNING", choices=levels)
    args = parser.parse_args()
    return args


def main():
    """Main entry point of the program."""
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    # Instantiate the ASRDataset
    dataset = ASRDataset(data_dir=args.dataset)

    # Load the STT model:
    model = whisper.load_model(args.model)

    # Load the text normalizer
    normalizer = BasicTextNormalizer()

    # Parse transcription and LM options:
    transcribe_options = parse_transcribe_options(args)
    set_lm_options(args)

    print(f"Transcribe options: {transcribe_options}")

    # Iterate through the dataset
    total_measures = defaultdict(list)
    for audio_path, label_text in tqdm(dataset):
        # Transcribe the audio:
        predicted_text = model.transcribe(audio_path, **transcribe_options)[
            "text"
        ]  # noqa: E501

        # Normalize text if required:
        if args.normalize_text:
            label_text = normalize_text(label_text, normalizer)
            predicted_text = normalize_text(predicted_text, normalizer)
        if args.normalize_diacritics:
            label_text = normalize_diacritics(label_text)
            predicted_text = normalize_diacritics(predicted_text)
        
        print("Frase ")
        print(predicted_text)

        # Calculate the metrics:
        measures = jiwer.compute_measures(label_text, predicted_text)
        measures["cer"] = jiwer.cer(label_text, predicted_text)
        for name, score in measures.items():
            if isinstance(score, (float, int)):
                total_measures[name].append(score)

    print("Scores:")
    for name, score in total_measures.items():
        score = np.array(score)
        print(f"Average {name}: {score.mean()} ± {score.std()}")


if __name__ == "__main__":
    main()
