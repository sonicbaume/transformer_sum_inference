
import logging
import sys
import types
from argparse import ArgumentParser, Namespace
from typing import List, Union

import pytorch_lightning as pl
import torch
from spacy.lang.en import English
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from classifier import (
    LinearClassifier,
    SimpleLinearClassifier,
    TransformerEncoderClassifier,
)
from pooling import Pooling

logger = logging.getLogger(__name__)


try:
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    MODEL_CLASSES = tuple(MODEL_MAPPING_NAMES.keys())
except ImportError:
    logger.warning(
        "Could not import `MODEL_MAPPING_NAMES` from transformers because it is an old version."
    )

    MODEL_CLASSES = (
        tuple(
            "Note: Only showing custom models because old version of `transformers` detected."
        )
    )


def longformer_modifier(final_dictionary):
    """
    Creates the ``global_attention_mask`` for the longformer. Tokens with global attention
    attend to all other tokens, and all other tokens attend to them. This is important for
    task-specific finetuning because it makes the model more flexible at representing the
    task. For example, for classification, the `<s>` token should be given global attention.
    For QA, all question tokens should also have global attention. For summarization,
    global attention is given to all of the `<s>` (RoBERTa 'CLS' equivalent) tokens. Please
    refer to the `Longformer paper <https://arxiv.org/abs/2004.05150>`_ for more details. Mask
    values selected in ``[0, 1]``: ``0`` for local attention, ``1`` for global attention.
    """
    # `batch_size` is the number of attention masks (one mask per input sequence)
    batch_size = len(final_dictionary["attention_mask"])
    # `sequence_length` is the number of tokens for the first sequence in the batch
    sequence_length = len(final_dictionary["attention_mask"][0])
    # create `global_attention_mask` using the above details
    global_attention_mask = torch.tensor([[0] * sequence_length] * batch_size)
    # set the `sent_rep_token_ids` to 1, which is global attention
    for idx, items in enumerate(final_dictionary["sent_rep_token_ids"]):
        global_attention_mask[idx, items] = 1

    final_dictionary["global_attention_mask"] = global_attention_mask
    # The `global_attention_mask` is passed through the model's `forward`
    # function as `**kwargs`.
    return final_dictionary


class ExtractiveSummarizer(pl.LightningModule):
    """
    A machine learning model that extractively summarizes an input text by scoring the sentences.
    Main class that handles the data loading, initial processing, training/testing/validating setup,
    and contains the actual model.
    """

    def __init__(self, hparams, embedding_model_config=None, classifier_obj=None):
        super(ExtractiveSummarizer, self).__init__()

        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)

        # Set new parameters to defaults if they do not exist in the `hparams` Namespace
        hparams.gradient_checkpointing = getattr(
            hparams, "gradient_checkpointing", False
        )
        hparams.tokenizer_no_use_fast = getattr(hparams, "tokenizer_no_use_fast", False)
        hparams.data_type = getattr(hparams, "data_type", "none")

        self.save_hyperparameters(hparams)
        self.forward_modify_inputs_callback = None

        if not embedding_model_config:
            embedding_model_config = AutoConfig.from_pretrained(
                hparams.model_name_or_path,
                gradient_checkpointing=hparams.gradient_checkpointing
            )

        self.word_embedding_model = AutoModel.from_config(embedding_model_config)

        if (
            any(
                x in hparams.model_name_or_path
                for x in ["roberta", "distil", "longformer"]
            )
        ) and not hparams.no_use_token_type_ids:
            logger.warning(
                (
                    "You are using a %s model but did not set "
                    + "--no_use_token_type_ids. This model does not support `token_type_ids` so "
                    + "this option has been automatically enabled."
                ),
                hparams.model_type,
            )
            self.hparams.no_use_token_type_ids = True

        self.emd_model_frozen = False
        if hparams.num_frozen_steps > 0:
            self.emd_model_frozen = True
            self.freeze_web_model()

        if hparams.pooling_mode == "sent_rep_tokens":
            self.pooling_model = Pooling(
                sent_rep_tokens=True, mean_tokens=False, max_tokens=False
            )
        elif hparams.pooling_mode == "max_tokens":
            self.pooling_model = Pooling(
                sent_rep_tokens=False, mean_tokens=False, max_tokens=True
            )
        else:
            self.pooling_model = Pooling(
                sent_rep_tokens=False, mean_tokens=True, max_tokens=False
            )

        # if a classifier object was passed when creating this model then store that as the
        # `encoder`
        if classifier_obj:
            self.encoder = classifier_obj
        # otherwise create the classifier using the `hparams.classifier` parameter if available
        # if the `hparams.classifier` parameter is missing then create a `LinearClassifier`
        else:
            # returns `classifier` value if it exists, otherwise returns False
            classifier_exists = getattr(hparams, "classifier", False)
            if (not classifier_exists) or (hparams.classifier == "linear"):
                self.encoder = LinearClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                )
            elif hparams.classifier == "simple_linear":
                self.encoder = SimpleLinearClassifier(
                    self.word_embedding_model.config.hidden_size
                )
            elif hparams.classifier == "transformer":
                self.encoder = TransformerEncoderClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                    num_layers=hparams.classifier_transformer_num_layers,
                )
            elif hparams.classifier == "transformer_linear":
                linear = LinearClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                )
                self.encoder = TransformerEncoderClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                    num_layers=hparams.classifier_transformer_num_layers,
                    custom_reduction=linear,
                )
            else:
                logger.error(
                    "%s is not a valid value for `--classifier`. Exiting...",
                    hparams.classifier,
                )
                sys.exit(1)

        # Set `hparams.no_test_block_trigrams` to False if it does not exist,
        # otherwise set its value to itself, resulting in no change
        self.hparams.no_test_block_trigrams = getattr(
            hparams, "no_test_block_trigrams", False
        )

        # BCELoss: https://pytorch.org/docs/stable/nn.html#bceloss
        # `reduction` is "none" so the mean can be computed with padding ignored.
        # `nn.BCEWithLogitsLoss` (which combines a sigmoid layer and the BCELoss
        # in one single class) is used because it takes advantage of the log-sum-exp
        # trick for numerical stability. Padding values are 0 and if 0 is the input
        # to the sigmoid function the output will be 0.5. This will cause issues when
        # inputs with more padding will have higher loss values. To solve this, all
        # padding values are set to -9e3 as the last step of each encoder. The sigmoid
        # function transforms -9e3 to nearly 0, thus preserving the proper loss
        # calculation. See `compute_loss()` for more info.
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

        # Data
        self.processor = SentencesProcessor(name="main_processor")

        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name
            if hparams.tokenizer_name
            else hparams.model_name_or_path,
            use_fast=(not self.hparams.tokenizer_no_use_fast),
        )

        self.train_dataloader_object = None  # not created yet
        self.datasets = None
        self.pad_batch_collate = None
        self.global_step_tracker = None
        self.rouge_metrics = None
        self.rouge_scorer = None

    def forward(
        self,
        input_ids,
        attention_mask,
        sent_rep_mask=None,
        token_type_ids=None,
        sent_rep_token_ids=None,
        sent_lengths=None,
        sent_lengths_mask=None,
        **kwargs,
    ):
        r"""Model forward function. See the `60 minute bliz tutorial <https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html>`_
        if you are unsure what a forward function is.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
                `What are input IDs? <https://huggingface.co/transformers/glossary.html#input-ids>`_
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token
                indices. Mask values selected in ``[0, 1]``: ``1`` for tokens that are NOT
                MASKED, ``0`` for MASKED tokens. `What are attention masks? <https://huggingface.co/transformers/glossary.html#attention-mask>`_
            sent_rep_mask (torch.Tensor, optional): Indicates which numbers in ``sent_rep_token_ids``
                are actually the locations of sentence representation ids and which are padding.
                Defaults to None.
            token_type_ids (torch.Tensor, optional): Usually, segment token indices to indicate
                first and second portions of the inputs. However, for summarization they are used
                to indicate different sentences. Depending on the size of the token type id vocabulary,
                these values may alternate between ``0`` and ``1`` or they may increase sequentially
                for each sentence in the input.. Defaults to None.
            sent_rep_token_ids (torch.Tensor, optional): The locations of the sentence representation
                tokens. Defaults to None.
            sent_lengths (torch.Tensor, optional):  A list of the lengths of each sentence in
                ``input_ids``. See :meth:`data.pad_batch_collate` for more info about the
                generation of thisfeature. Defaults to None.
            sent_lengths_mask (torch.Tensor, optional): Created on-the-fly by :meth:`data.pad_batch_collate`.
                Similar to ``sent_rep_mask``: ``1`` for value and ``0`` for padding. See
                :meth:`data.pad_batch_collate` for more info about the generation of this
                feature. Defaults to None.

        Returns:
            tuple: Contains the sentence scores and mask as ``torch.Tensor``\ s. The mask is either
            the ``sent_rep_mask`` or ``sent_lengths_mask`` depending on the pooling mode used
            during model initialization.
        """  # noqa: E501
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if not self.hparams.no_use_token_type_ids:
            inputs["token_type_ids"] = token_type_ids

        if self.forward_modify_inputs_callback:
            inputs = self.forward_modify_inputs_callback(inputs)  # skipcq: PYL-E1102

        outputs = self.word_embedding_model(**inputs, **kwargs)
        word_vectors = outputs[0]

        sents_vec, mask = self.pooling_model(
            word_vectors=word_vectors,
            sent_rep_token_ids=sent_rep_token_ids,
            sent_rep_mask=sent_rep_mask,
            sent_lengths=sent_lengths,
            sent_lengths_mask=sent_lengths_mask,
        )

        sent_scores = self.encoder(sents_vec, mask)
        return sent_scores, mask

    def unfreeze_web_model(self):
        """Un-freezes the ``word_embedding_model``"""
        for param in self.word_embedding_model.parameters():
            param.requires_grad = True

    def freeze_web_model(self):
        """Freezes the encoder ``word_embedding_model``"""
        for param in self.word_embedding_model.parameters():
            param.requires_grad = False


    def predict_sentences(
        self,
        input_sentences: Union[List[str], types.GeneratorType],
        raw_scores=False,
        num_summary_sentences=3,
        tokenized=False,
    ):
        """Summarizes ``input_sentences`` using the model.

        Args:
            input_sentences (list or generator): The sentences to be summarized as a
                list or a generator of spacy Spans (``spacy.tokens.span.Span``), which
                can be obtained by running ``nlp("input document").sents`` where
                ``nlp`` is a spacy model with a sentencizer.
            raw_scores (bool, optional): Return a list containing each sentence
                and its corespoding score instead of the summary. Defaults to False.
            num_summary_sentences (int, optional): The number of sentences in the
                output summary. This value specifies the number of top sentences to
                select as the summary. Defaults to 3.
            tokenized (bool, optional): If the input sentences are already tokenized
                using spacy. If true, ``input_sentences`` should be a list of lists
                where the outer list contains sentences and the inner lists contain
                tokens. Defaults to False.

        Returns:
            str: The summary text. If ``raw_scores`` is set then returns a list
            of input sentences and their corespoding scores.
        """
        # Create source text.
        # Don't add periods when joining because that creates a space before the period.
        if tokenized:
            src_txt = [
                " ".join([token.text for token in sentence if str(token) != "."]) + "."
                for sentence in input_sentences
            ]
        else:
            nlp = English()
            sentencizer = nlp.create_pipe("sentencizer")
            try:
                nlp.add_pipe(sentencizer)
            except ValueError as e:
                if e.args[0].startswith("[E966]"):
                    nlp.add_pipe("sentencizer")
                else:
                    raise e


            src_txt = [
                " ".join([token.text for token in nlp(sentence) if str(token) != "."])
                + "."
                for sentence in input_sentences
            ]

        input_ids = SentencesProcessor.get_input_ids(
            self.tokenizer,
            src_txt,
            sep_token=self.tokenizer.sep_token,
            cls_token=self.tokenizer.cls_token,
            bert_compatible_cls=True,
        )

        input_ids = torch.tensor(input_ids)
        attention_mask = [1] * len(input_ids)
        attention_mask = torch.tensor(attention_mask)

        sent_rep_token_ids = [
            i for i, t in enumerate(input_ids) if t == self.tokenizer.cls_token_id
        ]
        sent_rep_mask = torch.tensor([1] * len(sent_rep_token_ids))

        input_ids.unsqueeze_(0)
        attention_mask.unsqueeze_(0)
        sent_rep_mask.unsqueeze_(0)

        self.eval()

        with torch.no_grad():
            outputs, _ = self.forward(
                input_ids,
                attention_mask,
                sent_rep_mask=sent_rep_mask,
                sent_rep_token_ids=sent_rep_token_ids,
            )
            outputs = torch.sigmoid(outputs)

        if raw_scores:
            # key=sentence
            # value=score
            sent_scores = list(zip(src_txt, outputs.tolist()[0]))
            return sent_scores

        sorted_ids = (
            torch.argsort(outputs, dim=1, descending=True).detach().cpu().numpy()
        )
        logger.debug("Sorted sentence ids: %s", sorted_ids)
        selected_ids = sorted_ids[0, :num_summary_sentences]
        logger.debug("Selected sentence ids: %s", selected_ids)

        selected_sents = []
        selected_ids.sort()
        for i in selected_ids:
            selected_sents.append(src_txt[i])

        return selected_sents

    def predict(self, input_text: str, raw_scores=False, num_summary_sentences=3):
        """Summarizes ``input_text`` using the model.

        Args:
            input_text (str): The text to be summarized.
            raw_scores (bool, optional): Return a list containing each sentence
                and its corespoding score instead of the summary. Defaults to False.
            num_summary_sentences (int, optional): The number of sentences in the
                output summary. This value specifies the number of top sentences to
                select as the summary. Defaults to 3.

        Returns:
            str: The summary text. If ``raw_scores`` is set then returns a list
            of input sentences and their corespoding scores.
        """
        nlp = English()
        nlp.add_pipe("sentencizer")
        doc = nlp(input_text)

        return self.predict_sentences(
            input_sentences=doc.sents,
            raw_scores=raw_scores,
            num_summary_sentences=num_summary_sentences,
            tokenized=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Arguments specific to this model"""
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="bert-base-uncased",
            help="Path to pre-trained model or shortcut name. A list of shortcut names can be "
            + "found at https://huggingface.co/transformers/pretrained_models.html. "
            + "Community-uploaded models are located at https://huggingface.co/models.",
        )
        parser.add_argument(
            "--model_type",
            type=str,
            default="bert",
            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
        )
        parser.add_argument("--tokenizer_name", type=str, default="")
        parser.add_argument(
            "--tokenizer_no_use_fast",
            action="store_true",
            help="Don't use the fast version of the tokenizer for the specified model. "
            + "More info: https://huggingface.co/transformers/main_classes/tokenizer.html.",
        )
        parser.add_argument(
            "--max_seq_length",
            type=int,
            default=0,
            help="The maximum sequence length of processed documents.",
        )
        parser.add_argument(
            "--data_path", type=str, help="Directory containing the dataset."
        )
        parser.add_argument(
            "--data_type",
            default="none",
            type=str,
            choices=["txt", "pt", "none"],
            help="""The file extension of the prepared data. The 'map' `--dataloader_type`
            requires `txt` and the 'iterable' `--dataloader_type` works with both. If the data
            is not prepared yet (in JSON format) this value specifies the output format
            after processing. If the data is prepared, this value specifies the format to load.
            If it is `none` then the type of data to be loaded will be inferred from the
            `data_path`. If data needs to be prepared, this cannot be set to `none`.""",
        )
        parser.add_argument("--num_threads", type=int, default=4)
        parser.add_argument("--processing_num_threads", type=int, default=2)
        parser.add_argument(
            "--pooling_mode",
            type=str,
            default="sent_rep_tokens",
            choices=["sent_rep_tokens", "mean_tokens", "max_tokens"],
            help="How word vectors should be converted to sentence embeddings.",
        )
        parser.add_argument(
            "--num_frozen_steps",
            type=int,
            default=0,
            help="Freeze (don't train) the word embedding model for this many steps.",
        )
        parser.add_argument(
            "--batch_size",
            default=8,
            type=int,
            help="Batch size per GPU/CPU for training/evaluation/testing.",
        )
        parser.add_argument(
            "--dataloader_type",
            default="map",
            type=str,
            choices=["map", "iterable"],
            help="The style of dataloader to use. `map` is faster and uses less memory.",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            default=4,
            type=int,
            help="""The number of workers to use when loading data. A general place to
            start is to set num_workers equal to the number of CPU cores on your machine.
            If `--dataloader_type` is 'iterable' then this setting has no effect and
        num_workers will be 1. More details here: https://pytorch-lightning.readthedocs.io/en/latest/performance.html#num-workers""",  # noqa: E501
        )
        parser.add_argument(
            "--processor_no_bert_compatible_cls",
            action="store_false",
            help="If model uses bert compatible [CLS] tokens for sentence representations.",
        )
        parser.add_argument(
            "--only_preprocess",
            action="store_true",
            help="""Only preprocess and write the data to disk. Don't train model.
            This will force data to be preprocessed, even if it was already computed and
            is detected on disk, and any previous processed files will be overwritten.""",
        )
        parser.add_argument(
            "--preprocess_resume",
            action="store_true",
            help="Resume preprocessing. `--only_preprocess` must be set in order to resume. "
            + "Determines which files to process by finding the shards that do not have a "
            + 'coresponding ".pt" file in the data directory.',
        )
        parser.add_argument(
            "--create_token_type_ids",
            type=str,
            choices=["binary", "sequential"],
            default="binary",
            help="Create token type ids during preprocessing.",
        )
        parser.add_argument(
            "--no_use_token_type_ids",
            action="store_true",
            help="Set to not train with `token_type_ids` (don't pass them into the model).",
        )
        parser.add_argument(
            "--classifier",
            type=str,
            choices=["linear", "simple_linear", "transformer", "transformer_linear"],
            default="simple_linear",
            help="""Which classifier/encoder to use to reduce the hidden dimension of the sentence vectors.
                    `linear` - a `LinearClassifier` with two linear layers, dropout, and an activation function.
                    `simple_linear` - a `LinearClassifier` with one linear layer and a sigmoid.
                    `transformer` - a `TransformerEncoderClassifier` which runs the sentence vectors through some
                                    `nn.TransformerEncoderLayer`s and then a simple `nn.Linear` layer.
                    `transformer_linear` - a `TransformerEncoderClassifier` with a `LinearClassifier` as the
                                           `reduction` parameter, which results in the same thing as the `transformer` option but with a
                                           `LinearClassifier` instead of a `nn.Linear` layer.""",  # noqa: E501
        )
        parser.add_argument(
            "--classifier_dropout",
            type=float,
            default=0.1,
            help="The value for the dropout layers in the classifier.",
        )
        parser.add_argument(
            "--classifier_transformer_num_layers",
            type=int,
            default=2,
            help="The number of layers for the `transformer` classifier. Only has an effect if "
            + '`--classifier` contains "transformer".',
        )
        parser.add_argument(
            "--train_name",
            type=str,
            default="train",
            help="name for set of training files on disk (for loading and saving)",
        )
        parser.add_argument(
            "--val_name",
            type=str,
            default="val",
            help="name for set of validation files on disk (for loading and saving)",
        )
        parser.add_argument(
            "--test_name",
            type=str,
            default="test",
            help="name for set of testing files on disk (for loading and saving)",
        )
        parser.add_argument(
            "--test_id_method",
            type=str,
            default="top_k",
            choices=["greater_k", "top_k"],
            help="How to chose the top predictions from the model for ROUGE scores.",
        )
        parser.add_argument(
            "--test_k",
            type=float,
            default=3,
            help="The `k` parameter for the `--test_id_method`. Must be set if using the "
            + "`greater_k` option. (default: 3)",
        )
        parser.add_argument(
            "--no_test_block_trigrams",
            action="store_true",
            help="Disable trigram blocking when calculating ROUGE scores during testing. "
            + "This will increase repetition and thus decrease accuracy.",
        )
        parser.add_argument(
            "--test_use_pyrouge",
            action="store_true",
            help="""Use `pyrouge`, which is an interface to the official ROUGE software, instead of
            the pure-python implementation provided by `rouge-score`. You must have the real ROUGE
            package installed. More details about ROUGE 1.5.5 here: https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5.
            It is recommended to use this option for official scores. The `ROUGE-L` measurements
            from `pyrouge` are equivalent to the `rougeLsum` measurements from the default
            `rouge-score` package.""",  # noqa: E501
        )
        parser.add_argument(
            "--loss_key",
            type=str,
            choices=[
                "loss_total",
                "loss_total_norm_batch",
                "loss_avg_seq_sum",
                "loss_avg_seq_mean",
                "loss_avg",
            ],
            default="loss_avg_seq_mean",
            help="Which reduction method to use with BCELoss. See the "
            + "`experiments/loss_functions/` folder for info on how the default "
            + "(`loss_avg_seq_mean`) was chosen.",
        )
        return parser


class SentencesProcessor:
    r"""Create a `SentencesProcessor`

    Arguments:
        name (str, optional): A label for the ``SentencesProcessor`` object, used internally for
            saving if a save name is not specified in :meth:`data.SentencesProcessor.get_features`,
            Default is None.
        labels (list, optional): The label that goes with each sample, can be a list of lists where
            the inside lists are the labels for each sentence in the coresponding
            example. Default is None.
        examples (list, optional): List of ``InputExample``\ s. Default is None.
        verbose (bool, optional): Log extra information (such as examples of processed data
            points). Default is False.
    """

    def __init__(self, name=None, labels=None, examples=None, verbose=False):
        self.name = name
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    @classmethod
    def get_input_ids(
        cls,
        tokenizer,
        src_txt,
        bert_compatible_cls=True,
        sep_token=None,
        cls_token=None,
        max_length=None,
    ):
        """
        Get ``input_ids`` from ``src_txt`` using ``tokenizer``. See
        :meth:`~data.SentencesProcessor.get_features` for more info.
        """
        sep_token = str(sep_token)
        cls_token = str(cls_token)
        if max_length is None:
            try:
                max_length = list(tokenizer.max_model_input_sizes.values())[0]
            except AttributeError:
                max_length = tokenizer.model_max_length

        if max_length > 1_000_000:
            logger.warning(
                "Tokenizer maximum length is greater than 1,000,000. This is likely a mistake. "
                + "Resetting to 512 tokens."
            )
            max_length = 512

        # adds a '[CLS]' token between each sentence and outputs `input_ids`
        if bert_compatible_cls:
            # If the CLS or SEP tokens exist in the document as part of the dataset, then
            # set them to UNK
            unk_token = str(tokenizer.unk_token)
            src_txt = [
                sent.replace(sep_token, unk_token).replace(cls_token, unk_token)
                for sent in src_txt
            ]

            if not len(src_txt) < 2:  # if there is NOT 1 sentence
                # separate each sentence with ' [SEP] [CLS] ' (or model equivalent tokens) and
                # convert to string
                separation_string = " " + sep_token + " " + cls_token + " "
                text = separation_string.join(src_txt)
            else:
                try:
                    text = src_txt[0]
                except IndexError:
                    text = src_txt

            # tokenize
            src_subtokens = tokenizer.tokenize(text)
            # select first `(max_length-2)` tokens (so the following line of tokens can be added)
            src_subtokens = src_subtokens[: (max_length - 2)]
            # Insert '[CLS]' at beginning and append '[SEP]' to end (or model equivalent tokens)
            src_subtokens.insert(0, cls_token)
            src_subtokens.append(sep_token)
            # create `input_ids`
            input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
        else:
            input_ids = tokenizer.encode(
                src_txt,
                add_special_tokens=True,
                max_length=min(max_length, tokenizer.max_len),
            )

        return input_ids