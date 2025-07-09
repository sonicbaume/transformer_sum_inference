
import logging
import sys
import types
from argparse import Namespace
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
        return_idxs=False,
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

        # make seperate function for scoring
        # if raw_scores:
        #     # key=sentence
        #     # value=score
        #     sent_scores = list(zip(src_txt, outputs.tolist()[0]))
        #     return sent_scores

        sorted_ids = (
            torch.argsort(outputs, dim=1, descending=True).detach().cpu().numpy()
        )
        logger.debug("Sorted sentence ids: %s", sorted_ids)
        selected_ids = sorted_ids[0, :num_summary_sentences]
        logger.debug("Selected sentence ids: %s", selected_ids)

        selected_ids.sort()
        if return_idxs:
            return selected_ids.tolist()
        
        selected_sents = []
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