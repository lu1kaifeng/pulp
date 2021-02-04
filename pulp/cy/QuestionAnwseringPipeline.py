import csv
import json
import os
import pickle
import sys
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import chain
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from uuid import UUID

import numpy as np
import torch
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING, logger, AutoModelForQuestionAnswering

from transformers.configuration_auto import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.data import SquadExample
from .squad_examples_to_features import squad_convert_examples_to_features
from transformers.file_utils import add_end_docstrings, is_tf_available, is_torch_available
from transformers.modelcard import ModelCard
from transformers.pipelines import _ScikitCompat, ArgumentHandler, Pipeline, QuestionAnsweringArgumentHandler
from transformers.tokenization_auto import AutoTokenizer
from transformers.tokenization_bert import BasicTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.utils import logging


class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using any :obj:`ModelForQuestionAnswering`. See the
    `question answering examples <../task_summary.html#question-answering>`__ for more information.

    This question answering pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=question-answering>`__.
    """

    default_input_names = "question,context"

    def __init__(
            self,
            model: Union["PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            device: int = -1,
            task: str = "",
            **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=QuestionAnsweringArgumentHandler(),
            device=device,
            task=task,
            **kwargs,
        )

        self.check_model_type(
             MODEL_FOR_QUESTION_ANSWERING_MAPPING
        )

    @staticmethod
    def create_sample(
            question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the :class:`~transformers.SquadExample` internally.
        This helper method encapsulate all the logic for converting question(s) and context(s) to
        :class:`~transformers.SquadExample`.

        We currently support extractive question answering.

        Arguments:
            question (:obj:`str` or :obj:`List[str]`): The question(s) asked.
            context (:obj:`str` or :obj:`List[str]`): The context(s) in which we will look for the answer.

        Returns:
            One or a list of :class:`~transformers.SquadExample`: The corresponding
            :class:`~transformers.SquadExample` grouping question and context.
        """
        if isinstance(question, list):
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)

    def __call__(self, *args, **kwargs):
        """
        Answer the question(s) given as inputs by using the context(s).

        Args:
            args (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`):
                One or several :class:`~transformers.SquadExample` containing the question and context.
            X (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`, `optional`):
                One or several :class:`~transformers.SquadExample` containing the question and context
                (will be treated the same way as if passed as the first positional argument).
            data (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`, `optional`):
                One or several :class:`~transformers.SquadExample` containing the question and context
                (will be treated the same way as if passed as the first positional argument).
            question (:obj:`str` or :obj:`List[str]`):
                One or several question(s) (must be used in conjunction with the :obj:`context` argument).
            context (:obj:`str` or :obj:`List[str]`):
                One or several context(s) associated with the qustion(s) (must be used in conjunction with the
                :obj:`question` argument).
            topk (:obj:`int`, `optional`, defaults to 1):
                The number of answers to return (will be chosen by order of likelihood).
            doc_stride (:obj:`int`, `optional`, defaults to 128):
                If the context is too long to fit with the question for the model, it will be split in several chunks
                with some overlap. This argument controls the size of that overlap.
            max_answer_len (:obj:`int`, `optional`, defaults to 15):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_seq_len (:obj:`int`, `optional`, defaults to 384):
                The maximum length of the total sentence (context + question) after tokenization. The context will be
                split in several chunks (using :obj:`doc_stride`) if needed.
            max_question_len (:obj:`int`, `optional`, defaults to 64):
                The maximum length of the question after tokenization. It will be truncated if needed.
            handle_impossible_answer (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not we accept impossible as an answer.

        Return:
            A :obj:`dict` or a list of :obj:`dict`: Each result comes as a dictionary with the
            following keys:

            - **score** (:obj:`float`) -- The probability associated to the answer.
            - **start** (:obj:`int`) -- The start index of the answer (in the tokenized version of the input).
            - **end** (:obj:`int`) -- The end index of the answer (in the tokenized version of the input).
            - **answer** (:obj:`str`) -- The answer to the question.
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("max_seq_len", 384)
        kwargs.setdefault("max_question_len", 64)
        kwargs.setdefault("handle_impossible_answer", False)

        if kwargs["topk"] < 1:
            raise ValueError("topk parameter should be >= 1 (got {})".format(kwargs["topk"]))

        if kwargs["max_answer_len"] < 1:
            raise ValueError("max_answer_len parameter should be >= 1 (got {})".format(kwargs["max_answer_len"]))

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        features_list = [squad_convert_examples_to_features(
                examples=list(examples),
                tokenizer=self.tokenizer,
                max_seq_length=kwargs["max_seq_len"],
                doc_stride=kwargs["doc_stride"],
                max_query_length=kwargs["max_question_len"],
                padding_strategy=PaddingStrategy.MAX_LENGTH.value,
                is_training=False,
                tqdm_enabled=False,
            )]
        all_answers = []
        for features, example in zip(features_list, examples):
            model_input_names = self.tokenizer.model_input_names + ["input_ids"]
            fw_args = {k: [feature.__dict__[k] for feature in features] for k in model_input_names}

            # Manage tensor allocation on correct device
            with self.device_placement():
                if self.framework == "tf":
                    fw_args = {k: tf.constant(v) for (k, v) in fw_args.items()}
                    start, end = self.model(fw_args)[:2]
                    start, end = start.numpy(), end.numpy()
                else:
                    with torch.no_grad():
                        # Retrieve the score for the context tokens only (removing question tokens)
                        fw_args = {k: torch.tensor(v, device=self.device) for (k, v) in fw_args.items()}
                        start, end = self.model(**fw_args)[:2]
                        start, end = start.cpu().numpy(), end.cpu().numpy()

            min_null_score = 1000000  # large and positive
            answers = []
            for (feature, start_, end_) in zip(features, start, end):
                # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
                undesired_tokens = np.abs(np.array(feature.p_mask) - 1) & feature.attention_mask

                # Generate mask
                undesired_tokens_mask = undesired_tokens == 0.0

                # Make sure non-context indexes in the tensor cannot contribute to the softmax
                start_ = np.where(undesired_tokens_mask, -10000.0, start_)
                end_ = np.where(undesired_tokens_mask, -10000.0, end_)

                # Normalize logits and spans to retrieve the answer
                start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
                end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

                if kwargs["handle_impossible_answer"]:
                    min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

                # Mask CLS
                start_[0] = end_[0] = 0.0

                starts, ends, scores = self.decode(start_, end_, kwargs["topk"], kwargs["max_answer_len"])
                char_to_word = np.array(example.char_to_word_offset)

                # Convert the answer (tokens) back to the original text
                answers += [
                    {
                        "score": score.item(),
                        "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                        "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                        "answer": " ".join(
                            example.doc_tokens[feature.token_to_orig_map[s]: feature.token_to_orig_map[e] + 1]
                        ),
                    }
                    for s, e, score in zip(starts, ends, scores)
                ]

            if kwargs["handle_impossible_answer"]:
                answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})

            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: kwargs["topk"]]
            all_answers += answers

        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers

    def decode(self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
        """
        Take the output of any :obj:`ModelForQuestionAnswering` and will generate probalities for each span to be
        the actual answer.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than
        max_answer_len or answer end position being before the starting position.
        The method supports output the k-best answer through the topk argument.

        Args:
            start (:obj:`np.ndarray`): Individual start probabilities for each token.
            end (:obj:`np.ndarray`): Individual end probabilities for each token.
            topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
        return start, end, candidates[0, start, end]

    def span_to_answer(self, text: str, start: int, end: int) -> Dict[str, Union[str, int]]:
        """
        When decoding from token probalities, this method maps token indexes to actual word in
        the initial context.

        Args:
            text (:obj:`str`): The actual context to extract the answer from.
            start (:obj:`int`): The answer starting token index.
            end (:obj:`int`): The answer end token index.

        Returns:
            Dictionary like :obj:`{'answer': str, 'start': int, 'end': int}`
        """
        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0

        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                if token_idx == end:
                    char_end_idx = chars_idx + len(word)

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }


def pipeline(
        task: str,
        model: Optional = None,
        config: Optional[Union[str, PretrainedConfig]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        framework: Optional[str] = None,
        **kwargs
) -> Pipeline:
    """
    Utility factory method to build a :class:`~transformers.Pipeline`.

    Pipelines are made of:

        - A :doc:`tokenizer <tokenizer>` in charge of mapping raw textual input to token.
        - A :doc:`model <model>` to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`: will return a :class:`~transformers.FeatureExtractionPipeline`.
            - :obj:`"sentiment-analysis"`: will return a :class:`~transformers.TextClassificationPipeline`.
            - :obj:`"ner"`: will return a :class:`~transformers.TokenClassificationPipeline`.
            - :obj:`"question-answering"`: will return a :class:`~transformers.QuestionAnsweringPipeline`.
            - :obj:`"fill-mask"`: will return a :class:`~transformers.FillMaskPipeline`.
            - :obj:`"summarization"`: will return a :class:`~transformers.SummarizationPipeline`.
            - :obj:`"translation_xx_to_yy"`: will return a :class:`~transformers.TranslationPipeline`.
            - :obj:`"text-generation"`: will return a :class:`~transformers.TextGenerationPipeline`.
            - :obj:`"conversation"`: will return a :class:`~transformers.ConversationalPipeline`.
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from :class:`~transformers.PreTrainedModel` (for PyTorch)
            or :class:`~transformers.TFPreTrainedModel` (for TensorFlow).

            If not provided, the default for the :obj:`task` will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If not provided, the default for the :obj:`task` will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.

            If not provided, the default for the :obj:`task` will be loaded.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no
            model is provided.
        kwargs:
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        :class:`~transformers.Pipeline`: A suitable pipeline for the task.

    Examples::

        >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        >>> # Sentiment analysis pipeline
        >>> pipeline('sentiment-analysis')

        >>> # Question answering pipeline, specifying the checkpoint identifier
        >>> pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

        >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
        >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> pipeline('ner', model=model, tokenizer=tokenizer)
    """

    framework = 'pt'

    task_class, model_class = QuestionAnsweringPipeline, AutoModelForQuestionAnswering

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = 'distilbert-base-cased-distilled-squad'

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        elif isinstance(config, str):
            tokenizer = config
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    modelcard = None
    # Try to infer modelcard from model or config name (if provided as str)
    if isinstance(model, str):
        modelcard = model
    elif isinstance(config, str):
        modelcard = config

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config)

    # Instantiate modelcard if needed
    if isinstance(modelcard, str):
        modelcard = ModelCard.from_pretrained(modelcard)

    # Instantiate model if needed
    if isinstance(model, str):
        # Handle transparent TF/PT model conversion
        model_kwargs = {}
        if framework == "pt" and model.endswith(".h5"):
            model_kwargs["from_tf"] = True
            logger.warning(
                "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                "Trying to load the model with PyTorch."
            )
        elif framework == "tf" and model.endswith(".bin"):
            model_kwargs["from_pt"] = True
            logger.warning(
                "Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. "
                "Trying to load the model with Tensorflow."
            )
        model = model_class.from_pretrained(model, config=config, **model_kwargs)

    return task_class(model=model, tokenizer=tokenizer, modelcard=modelcard, framework=framework, task=task, **kwargs)
