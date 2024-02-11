#!/usr/bin/env python
"""Evaluates a Whisper model in OpenAI format in a dataset (CV by default)."""

import sys
import logging
import argparse
from collections import defaultdict

import jiwer
import whisper  # pylint: disable=E0401
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import whisper_decoder_with_lm  # pylint: disable=unused-import # noqa: E501,F401
from whisper_decoder_with_lm import LMOptions


def parse_none(value):
    """Converts `"None"` string to `None` value in Python.

    Used to parse command line values.

    Parameters
    ----------
    value : str
        The input value.

    Returns
    -------
    str or None
        The output value with None parsed.
    """
    return None if value == "None" else value


def parse_args():
    """Parses command line arguments.

    Returns
    -------
    namespace
        The namespace populated with the command line argument values.
    """
    parser = argparse.ArgumentParser(
        description="Evaluates a Whipser model in OpenAI format."
    )
    parser.add_argument(
        "model",
        help="Path or name of the OpenAI model to load.",
    )
    parser.add_argument(
        "--audios",
        "-a",
        default=[],
        nargs="+",
        help="Transcribe a list of audios instead of using a dataset.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="mozilla-foundation/common_voice_13_0",
        help="Path or name of the Hugging Face dataset. Defaults to CV 13.",
    )
    parser.add_argument(
        "--dataset_name",
        "-dn",
        default="eu",
        help=(
            "Defining the name of the dataset configuration for Hugging Face. "
            "For Common Voice datasets, this represents the language. "
            "Defaults to `eu`."
        ),
    )
    parser.add_argument(
        "--dataset_split",
        "-ds",
        default="test",
        help="Which split of the data to load. Defaults to `test`.",
    )
    parser.add_argument(
        "--skip_normalize",
        "-n",
        action="store_true",
        help="Whether to normalize the text (enabled by default)",
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
    parser.add_argument("--log-level", "-l", default="INFO", choices=levels)
    args = parser.parse_args()
    return args


def parse_transcribe_options(args):
    """Gets the transcriptions options for the `Model.transcribe()`.

    Parameters
    ----------
    args : namespace or dict
        The namespace populated with the command line argument values.

    Returns
    -------
    dict
        Transcription options.
    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    # Decoding options:
    decode_options = {
        "temperature": args.get("temperature", 0.0),
        "best_of": args.get("best_of", None),
        "beam_size": args.get("beam_size", None),
        "patience": args.get("patience", None),
    }
    logging.info("Decode options: %s", decode_options)
    transcribe_options = {"task": "transcribe", **decode_options}
    return transcribe_options


def set_lm_options(args):
    """Sets LM options from the arguments.

    Parameters
    ----------
    arg : namespace
        The namespace populated with the command line argument values.
    """
    # LM options:
    if hasattr(args, "lm_alpha") and args.lm_alpha is not None:
        LMOptions().lm_alpha = args.lm_alpha
    if hasattr(args, "lm_beta") and args.lm_beta is not None:
        LMOptions().lm_beta = args.lm_beta
    if hasattr(args, "lm_eos") and args.lm_eos is not None:
        LMOptions().lm_eos = args.lm_eos
    if hasattr(args, "lm_normalize") and args.lm_normalize is not None:
        LMOptions().lm_normalize = args.lm_normalize
    if (
        hasattr(args, "lm_token_threshold")
        and args.lm_token_threshold is not None  # noqa: E501
    ):  # noqa: E501
        LMOptions().lm_token_threshold = args.lm_token_threshold
    if hasattr(args, "lm_path") and args.lm_path is not None:
        LMOptions().lm_path = args.lm_path
        logging.info("LM path: %s", LMOptions().lm_path)
        logging.info("LM alpha: %f", LMOptions().lm_alpha)
        logging.info("LM beta: %f", LMOptions().lm_beta)
        logging.info("LM eos: %s", LMOptions().lm_eos)
        logging.info("LM normalize: %s", LMOptions().lm_normalize)
        logging.info("LM token threshold: %s", LMOptions().lm_token_threshold)


def main():
    """Main entry point of the program."""
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    # Print the command line run:
    logging.info("Command: %s", " ".join(sys.argv))

    logging.info("Loading model: %s", args.model)
    model = whisper.load_model(args.model)

    logging.info("Loading dataset: %s", args.dataset)
    logging.info("- name: %s", args.dataset_name)
    logging.info("- split: %s", args.dataset_split)
    dataset = load_dataset(
        args.dataset,
        parse_none(args.dataset_name),
        split=args.dataset_split,
        token=True,
    )
    dataset = dataset.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )

    # Parse transcription and LM options:
    transcribe_options = parse_transcribe_options(args)
    set_lm_options(args)

    # Transcribe a list of audios:
    if len(args.audios) > 0:
        logging.info("Transcriptions:")
        for audio in args.audios:
            logging.debug("Transcribing audio: %s", audio)
            result = model.transcribe(audio, **transcribe_options)
            print("- " + audio + ":", result["text"])
        sys.exit(0)

    # Text normalizing function:
    if not args.skip_normalize:
        normalizer = BasicTextNormalizer()

    logging.info("Evaluating the dataset:")
    total_measures = defaultdict(list)
    for example in tqdm(dataset):
        # Transcribe the example:
        label_text = example["sentence"]
        predicted_text = model.transcribe(
            example["audio"]["path"], **transcribe_options
        )["text"]
        # Compute the score:
        if not args.skip_normalize:
            label_text = normalizer(label_text).strip().lower()
            predicted_text = normalizer(predicted_text).strip().lower()
        # Save the scores:
        measures = jiwer.compute_measures(label_text, predicted_text)
        measures["cer"] = jiwer.cer(label_text, predicted_text)
        for name, score in measures.items():
            if isinstance(score, (float, int)):
                total_measures[name].append(score)

    logging.info("Scores:")
    for name, score in total_measures.items():
        score = np.array(score)
        print(f"Average {name}: {score.mean()} ± {score.std()}")


if __name__ == "__main__":
    main()
