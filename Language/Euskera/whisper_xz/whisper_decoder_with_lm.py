"""Extends the internal Whisper classes to support a KenLM.

This code is still used here, but has been recently moved to the following
whisper fork: https://github.com/zuazo-forks/whisper/tree/lm-simple
"""

import string
import logging
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from whisper import Whisper
from whisper.tokenizer import Tokenizer
from whisper.decoding import (
    BeamSearchDecoder,
    DecodingOptions,
    DecodingTask,
    Inference,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import kenlm


# Extending the DecodingOptions class to support an LM
# ====================================================


class LMOptions:  # pylint: disable=too-few-public-methods
    """Singleton class to pass the LM options to the Beam Search algorithm.

    I did not found a better way to pass the configuration options to the
    `BeamSearchDecoderWithLM` class.
    """

    _instance = None

    # A KenLM n-gram language model path:
    lm_path: str = None

    # The maximum of the alpha hyperparameter of the CTC decoder explored
    # during hyperparameter optimization. Language Model weight.
    lm_alpha: float = 0.931289039105002

    # End of string character list for the LM:
    lm_eos: str = "!?."

    # The maximum beta hyperparameter of the CTC decoder explored during
    # hyperparameter optimization. Word insertion weight.
    lm_beta: float = 1.1834137581510284

    # Whether to normalize text before sending it to the languge model:
    lm_normalize: bool = True

    # Minimum number of tokens in a sequence required before applying language
    # model scoring. This prevents premature evaluation on short sequences.
    lm_token_threshold: int = 4

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LMOptions, cls).__new__(cls)
        return cls._instance


# New Beam Search class with LM support (KenLM)
# =============================================


class BeamSearchDecoderWithLM(
    BeamSearchDecoder
):  # pylint: disable=too-many-instance-attributes
    """New Beam Search class with LM support (KenLM)."""

    def __init__(
        self,
        beam_size: int,
        tokenizer: Tokenizer,
        inference: Inference,
        patience: Optional[float] = None,
        lm_path: Optional[str] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
        lm_eos: Optional[str] = None,
        lm_normalize: Optional[bool] = True,
    ):  # pylint: disable=too-many-arguments
        super().__init__(beam_size, tokenizer.eot, inference, patience)
        self.tokenizer = tokenizer
        self.special_tokens = list(self.tokenizer.special_tokens.values())
        self.lm_model = (
            kenlm.Model(lm_path) if lm_path is not None else None
        )  # pylint: disable=c-extension-no-member
        self.lm_alpha = lm_alpha or 0.0
        self.lm_beta = lm_beta or 0.0
        self.lm_eos = lm_eos or ""  # end of sentence chars
        self.lm_eow = set(string.punctuation)  # end of word chars
        self.lm_normalize = lm_normalize  # whether to normalize the LM text
        self.lm_normalizer = BasicTextNormalizer()  # normalize for the KenLM
        self.finished_sequences = None

    def lm_score_and_word_count(self, sequence) -> Tuple[float, int]:
        """Get language model score and word count for a sequence.

        Parameters
        ----------
        sequence : tuple of int
            A sequence of token IDs.

        Returns
        -------
        float
            The language model score for the decoded text of the sequence.
        int
            The number of words in the decoded text of the sequence.
        """
        if not self.lm_model:
            return None, 0.0

        # Convert sequence of tokens to text
        sequence = tuple(t for t in sequence if t not in self.special_tokens)
        if len(sequence) < LMOptions().lm_token_threshold:
            return None, 0.0
        text = self.tokenizer.decode(sequence)

        # Early return for empty text
        if not text:
            return None, 0.0
        logging.debug('LM text: "%s"', text)

        # Normalize the text
        if self.lm_normalize:
            normalized_text = self.lm_normalizer(text)
        else:
            normalized_text = text
        logging.debug('LM text normalized: "%s"', normalized_text)

        # Check for end of sentence and end of word:
        eos = text[-1] in self.lm_eos

        word_count = len(normalized_text.split())
        logging.debug("Word count: %d", word_count)

        # In KenLM, the most probable sequences have a higher score:
        score = self.lm_model.score(normalized_text, bos=True, eos=eos)
        logging.debug("LM score: %f", score)

        return score, word_count

    def update(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements # noqa: E501
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Updates the beam search state with language model scoring.

        This method performs a beam search step and updates internal states,
        such as finished sequences and token caches. The beam search step
        includes LM scoring for ranking beam candidates.

        The method internally:

        1. Calculates the cumulative log probabilities for potential beam
           candidates by considering both the model's predictions and optional
           LM scores.
        2. Ranks the candidates and keeps the top 'beam_size' sequences for
           each audio sample.
        3. Checks and keeps track of sequences that have finished decoding.

        This code is based on `BeamSearchDecoder.update()`, but with the
        additional integration of language model scoring.

        Parameters
        ----------
        tokens : Tensor)
            Current tokens in the beam. Should have shape
            [n_audio * beam_size, seq_len], where n_audio is the number of
            audio samples and beam_size is the number of beams.
        logits : Tensor
            Raw prediction scores for the next token, of shape
            [n_audio * beam_size, vocab_size].
        sum_logprobs : Tensor
            Cumulative log probabilities of the sequences in the beam so far.
            Should have shape [n_audio * beam_size].

        Returns
        -------
        Tuple[Tensor, bool]:
            - A tensor with the updated tokens for each beam, of shape
              [n_audio * beam_size, seq_len].
            - A boolean indicating if the beam search is completed for all
              audio samples.

        Raises
        ------
        ValueError:
            If the tokens tensor's shape is not divisible by the beam size.
        """
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible
            # candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(
                    *logprobs[idx].topk(self.beam_size + 1)
                ):  # noqa: E501
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    logging.debug("AC score (new_logprob): %f", new_logprob)
                    sequence = tuple(prefix + [token.item()])
                    # Adjust the score by adding the LM score:
                    lm_score, wordc = self.lm_score_and_word_count(sequence)
                    if lm_score is not None:  # if it is a word boundary
                        lm_adjusted_score = (
                            new_logprob
                            + self.lm_alpha * lm_score
                            + wordc * self.lm_beta
                        )
                        scores[sequence] = lm_adjusted_score
                    else:
                        scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences
            # for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(  # pylint: disable=no-member
            next_tokens, device=tokens.device
        )  # pylint: disable=no-member
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(
                newly_finished, key=newly_finished.get, reverse=True
            ):  # noqa: E501
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed


# Extending the DecodingTask class to support an BeamSearchWithLM
# ===============================================================


# Store a reference to the original __init__
original_decoding_task_init = DecodingTask.__init__


def new_decoding_task_init(self, model: Whisper, options: DecodingOptions):
    """New constructor for the DecodingTask class.

    Example
    -------
    >>> DecodingTask.__init__ = new_decoding_task_init
    """
    # Call the original constructor using the stored reference:
    original_decoding_task_init(self, model, options)

    # New logic:
    lm_options = LMOptions()
    if options.beam_size is not None and lm_options.lm_path is not None:
        self.decoder = BeamSearchDecoderWithLM(
            options.beam_size,
            self.tokenizer,
            self.inference,
            options.patience,
            lm_options.lm_path,
            lm_options.lm_alpha,
            lm_options.lm_beta,
            lm_options.lm_eos,
            lm_options.lm_normalize,
        )


# Monkey patching the DecodingTask constructor:
DecodingTask.__init__ = new_decoding_task_init
