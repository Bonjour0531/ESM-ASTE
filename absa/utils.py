import json
import collections
import numpy as np
import pickle

import bert.tokenization as tokenization
from squad.squad_utils import  get_final_text, _get_best_indexes
from squad.squad_evaluate import exact_match_score, f1_score

label_to_id = {'other': 0, 'neutral': 1, 'positive': 2, 'negative': 3, 'conflict': 4}
id_to_label = {0: 'other', 1: 'neutral', 2: 'positive', 3: 'negative', 4: 'conflict'}

label_map = {'O':0, 'B-aspect':1, 'I-aspect':2, 'E-aspect':3, 'S-aspect':4,
             'B-opinion':5, 'I-opinion':6, 'E-opinion':7, 'S-opinion':8 }

id2sent = {0: 'NEU', 1: 'POS', 2: 'NEG', 3: 'NONE'}
sent2id = {'NEU':0, 'POS':1, 'NEG':2, 'NONE':3}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputFeaturesWithlc(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, aspect_mask, opinion_mask, local_context_mask, aspect_length, opinion_length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.aspect_mask = aspect_mask
        self.opinion_mask = opinion_mask
        self.local_context_mask = local_context_mask
        self.aspect_length = aspect_length
        self.opinion_length = opinion_length

class InputClsExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, aspect, opinion, label=None):
        self.guid = guid
        self.text = text
        self.aspect = aspect
        self.opinion = opinion
        self.label = label

class InputClsLcExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, aspect, opinion, aspect_span, opinion_span, local_context_span,
                            aspect_length, opinion_length, label=None):
        self.guid = guid
        self.text = text
        self.aspect = aspect
        self.opinion = opinion
        self.label = label
        self.aspect_span = aspect_span
        self.opinion_span = opinion_span
        self.local_context_span = local_context_span
        self.aspect_length = aspect_length
        self.opinion_length = opinion_length

class InputPipelineClsLcExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, example_id, text, aspect, opinion, aspect_span, opinion_span, local_context_span,
                            aspect_length, opinion_length, label=None, gold_triplets=None):
        self.guid = guid
        self.example_id = example_id
        self.text = text
        self.aspect = aspect
        self.opinion = opinion
        self.label = label
        self.gold_triplets = gold_triplets
        self.aspect_span = aspect_span
        self.opinion_span = opinion_span
        self.local_context_span = local_context_span
        self.aspect_length = aspect_length
        self.opinion_length = opinion_length

class InputPipelineClsExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, example_id, text, aspect, opinion, label=None, gold_triplets=None):
        self.guid = guid
        self.example_id = example_id
        self.text = text
        self.aspect = aspect
        self.opinion = opinion
        self.label = label
        self.gold_triplets = gold_triplets

class SemEvalExample(object):
    def __init__(self,
                 example_id,
                 sent_tokens,
                 term_texts=None,
                 start_positions=None,
                 end_positions=None,
                 polarities=None):
        self.example_id = example_id
        self.sent_tokens = sent_tokens
        self.term_texts = term_texts
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.polarities = polarities

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        # s += "example_id: %s" % (tokenization.printable_text(self.example_id))
        s += ", sent_tokens: [%s]" % (" ".join(self.sent_tokens))
        if self.term_texts:
            s += ", term_texts: {}".format(self.term_texts)
        # if self.start_positions:
        #     s += ", start_positions: {}".format(self.start_positions)
        # if self.end_positions:
        #     s += ", end_positions: {}".format(self.end_positions)
        if self.polarities:
            s += ", polarities: {}".format(self.polarities)
        return s


class InputFeaturesori(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_positions=None,
                 end_positions=None,
                 start_indexes=None,
                 end_indexes=None,
                 bio_labels=None,
                 polarity_positions=None,
                 polarity_labels=None,
                 label_masks=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.start_indexes = start_indexes
        self.end_indexes = end_indexes
        self.bio_labels = bio_labels
        self.polarity_positions = polarity_positions
        self.polarity_labels = polarity_labels
        self.label_masks = label_masks


def convert_examples_to_features_ori(examples, tokenizer, max_seq_length, verbose_logging=False, logger=None):
    max_term_num = max([len(example.term_texts) for (example_index, example) in enumerate(examples)])
    max_sent_length, max_term_length = 0, 0

    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(examples):
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.sent_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        if len(all_doc_tokens) > max_sent_length:
            max_sent_length = len(all_doc_tokens)

        tok_start_positions = []
        tok_end_positions = []
        for start_position, end_position in \
                zip(example.start_positions, example.end_positions):
            tok_start_position = orig_to_tok_index[start_position]
            if end_position < len(example.sent_tokens) - 1:
                tok_end_position = orig_to_tok_index[end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            tok_start_positions.append(tok_start_position)
            tok_end_positions.append(tok_end_position)

        # Account for [CLS] and [SEP] with "- 2"
        if len(all_doc_tokens) > max_seq_length - 2:
            all_doc_tokens = all_doc_tokens[0:(max_seq_length - 2)]

        tokens = []
        token_to_orig_map = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)

        for index, token in enumerate(all_doc_tokens):
            token_to_orig_map[len(tokens)] = tok_to_orig_index[index]
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # For distant supervision, we annotate the positions of all answer spans
        start_positions = [0] * len(input_ids)
        end_positions = [0] * len(input_ids)
        bio_labels = [0] * len(input_ids)
        polarity_positions = [0] * len(input_ids)
        start_indexes, end_indexes = [], []
        for tok_start_position, tok_end_position, polarity in zip(tok_start_positions, tok_end_positions, example.polarities):
            if (tok_start_position >= 0 and tok_end_position <= (max_seq_length - 1)):
                start_position = tok_start_position + 1   # [CLS]
                end_position = tok_end_position + 1   # [CLS]
                start_positions[start_position] = 1
                end_positions[end_position] = 1
                start_indexes.append(start_position)
                end_indexes.append(end_position)
                term_length = tok_end_position - tok_start_position + 1
                max_term_length = term_length if term_length > max_term_length else max_term_length
                bio_labels[start_position] = 1  # 'B'
                if start_position < end_position:
                    for idx in range(start_position + 1, end_position + 1):
                        bio_labels[idx] = 2  # 'I'
                for idx in range(start_position, end_position + 1):
                    polarity_positions[idx] = label_to_id[polarity]

        polarity_labels = [label_to_id[polarity] for polarity in example.polarities]
        label_masks = [1] * len(polarity_labels)

        while len(start_indexes) < max_term_num:
            start_indexes.append(0)
            end_indexes.append(0)
            polarity_labels.append(0)
            label_masks.append(0)

        assert len(start_indexes) == max_term_num
        assert len(end_indexes) == max_term_num
        assert len(polarity_labels) == max_term_num
        assert len(label_masks) == max_term_num

        if example_index < 1 and verbose_logging:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("tokens: {}".format(tokens))
            logger.info("token_to_orig_map: {}".format(token_to_orig_map))
            logger.info("start_indexes: {}".format(start_indexes))
            logger.info("end_indexes: {}".format(end_indexes))
            logger.info("bio_labels: {}".format(bio_labels))
            logger.info("polarity_positions: {}".format(polarity_positions))
            logger.info("polarity_labels: {}".format(polarity_labels))

        features.append(
            InputFeaturesori(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                start_indexes=start_indexes,
                end_indexes=end_indexes,
                bio_labels=bio_labels,
                polarity_positions=polarity_positions,
                polarity_labels=polarity_labels,
                label_masks=label_masks))
        unique_id += 1
    logger.info("Max sentence length: {}".format(max_sent_length))
    logger.info("Max term length: {}".format(max_term_length))
    logger.info("Max term num: {}".format(max_term_num))
    return features


def convert_examples_to_features(examples, tokenizer, max_seq_length, logger=None):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_org = example.text
        tokens_a = []
        label_ids = []
        for (i, token) in enumerate(tokens_a_org):
            sub_tokens = tokenizer.tokenize(token)
            # for sub_token in sub_tokens:
            if len(sub_tokens) != 0:
                tokens_a.append(sub_tokens[0])  # use the first token
                label_ids.append(label_map[example.label[i]])

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            label_ids = label_ids[:max_seq_length - 2]

        label_ids = [0] + label_ids + [0]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, str(label_ids)))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids))
    return features

def convert_examples_to_features_cls(examples, tokenizer, max_seq_length, logger=None, sep_tok=None):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_org = example.text
        aspect = example.aspect.split()
        opinion = example.opinion.split()
        label_id = example.label
        tokens_a = []
        for (i, token) in enumerate(tokens_a_org):
            sub_tokens = tokenizer.tokenize(token)
            # for sub_token in sub_tokens:
            if len(sub_tokens) != 0:
                tokens_a.append(sub_tokens[0])  # use the first token

        # 构造 cls,…, sep, aspect, sep, opinion, sep
        keep_count = 4 + len(aspect) + len(opinion)
        if len(tokens_a) > max_seq_length - keep_count:
            tokens_a = tokens_a[:(max_seq_length - keep_count)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        seg_len = len(tokens.copy())
        segment_ids = [0] * len(tokens)
        for token in aspect:
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) != 0: tokens.append(sub_tokens[0])  # use the first token
        if sep_tok:
            tokens.append(sep_tok)
        else:
            tokens.append("[SEP]")
        for token in opinion:
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) != 0: tokens.append(sub_tokens[0])  # use the first token
        tokens.append("[SEP]")

        segment_ids += [1] * (len(tokens)-seg_len)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def convert_examples_to_features_cls_withlc(examples, tokenizer, max_seq_length, logger=None, sep_tok=None):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_org = example.text
        aspect = example.aspect.split()
        opinion = example.opinion.split()
        label_id = example.label
        aspect_span = example.aspect_span
        opinion_span = example.opinion_span
        local_context_span = example.local_context_span
        aspect_length = example.aspect_length
        opinion_length = example.opinion_length

        tokens_a = []
        for (i, token) in enumerate(tokens_a_org):
            sub_tokens = tokenizer.tokenize(token)
            # for sub_token in sub_tokens:
            if len(sub_tokens) != 0:
                tokens_a.append(sub_tokens[0])  # use the first token

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        aspect_mask = [0] * len(tokens)
        aspect_mask[aspect_span[0]+1 : aspect_span[1]+2] = [1] * (aspect_span[1]+1 - aspect_span[0])
        opinion_mask = [0] * len(tokens)
        opinion_mask[opinion_span[0]+1 : opinion_span[1]+2] = [1] * (opinion_span[1]+1 - opinion_span[0])
        local_context_mask = [0] * len(tokens)
        if local_context_span[1] - local_context_span[0] > 1:
            local_context_mask[local_context_span[0]+2 : local_context_span[1]+1] = [1]*(local_context_span[1]-local_context_span[0]-1)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        aspect_mask += padding
        opinion_mask += padding
        local_context_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(opinion_mask) == max_seq_length
        assert len(aspect_mask) == max_seq_length
        assert len(local_context_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))
            logger.info("aspect_mask: %s" % " ".join([str(x) for x in aspect_mask]))
            logger.info("opinion_mask: %s" % " ".join([str(x) for x in opinion_mask]))
            logger.info("local_context_mask: %s" % " ".join([str(x) for x in local_context_mask]))
            logger.info("aspect_length: %s " % (str(aspect_length)))
            logger.info("opinion_length: %s " % (str(opinion_length)))


        features.append(
            InputFeaturesWithlc(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          aspect_mask = aspect_mask,
                          opinion_mask = opinion_mask,
                          local_context_mask=local_context_mask,
                          aspect_length=aspect_length,
                          opinion_length=opinion_length))
    return features

def convert_examples_to_features_cls_auxiliary(examples, tokenizer, max_seq_length, logger=None, sep_tok=None):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_org = example.text
        aspect = example.aspect.split()
        opinion = example.opinion.split()
        label_id = example.label
        tokens_a = []
        for (i, token) in enumerate(tokens_a_org):
            sub_tokens = tokenizer.tokenize(token)
            # for sub_token in sub_tokens:
            if len(sub_tokens) != 0:
                tokens_a.append(sub_tokens[0])  # use the first token

        # 构造 cls,…, sep, aspect-opinion auxiliary sentences, sep
        #  What is the sentiment relation between *aspect* and *opinion*
        # auxiliary_template = "What is the sentiment relation between"  ## 'and'
        # keep_count = 3 + len(aspect) + len(opinion) + len(auxiliary_template) + 1
        keep_count = 3 + len(aspect) + len(opinion)
        if len(tokens_a) > max_seq_length - keep_count:
            tokens_a = tokens_a[:(max_seq_length - keep_count)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        seg_len = len(tokens.copy())
        segment_ids = [0] * len(tokens)

        for token in aspect:
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) != 0: tokens.append(sub_tokens[0])  # use the first token

        tokens.append('-')

        for token in opinion:
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) != 0: tokens.append(sub_tokens[0])  # use the first token
        tokens.append("[SEP]")

        segment_ids += [1] * (len(tokens)-seg_len)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features
RawSpanResult = collections.namedtuple("RawSpanResult",
                                       ["unique_id", "start_logits", "end_logits"])

RawSpanCollapsedResult = collections.namedtuple("RawSpanCollapsedResult",
                                       ["unique_id", "neu_start_logits", "neu_end_logits", "pos_start_logits", "pos_end_logits",
                                        "neg_start_logits", "neg_end_logits"])

RawBIOResult = collections.namedtuple("RawBIOResult", ["unique_id", "bio_pred"])

RawBIOClsResult = collections.namedtuple("RawBIOClsResult", ["unique_id", "start_indexes", "end_indexes", "bio_pred", "span_masks"])

RawFinalResult = collections.namedtuple("RawFinalResult",
                                        ["unique_id", "start_indexes", "end_indexes", "cls_pred", "span_masks"])


def wrapped_get_final_text(example, feature, start_index, end_index, do_lower_case, verbose_logging, logger):
    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    orig_doc_start = feature.token_to_orig_map[start_index]
    orig_doc_end = feature.token_to_orig_map[end_index]
    orig_tokens = example.sent_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)

    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, logger)
    return final_text


def span_annotate_candidates(all_examples, batch_features, batch_results, filter_type, is_training, use_heuristics, use_nms,
                             logit_threshold, n_best_size, max_answer_length, do_lower_case, verbose_logging, logger):
    """Annotate top-k candidate answers into features."""
    unique_id_to_result = {}
    for result in batch_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    batch_span_starts, batch_span_ends, batch_labels, batch_label_masks = [], [], [], []
    for (feature_index, feature) in enumerate(batch_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        seen_predictions = {}
        span_starts, span_ends, labels, label_masks = [], [], [], []
        if is_training:
            # add ground-truth terms
            for start_index, end_index, polarity_label, mask in \
                    zip(feature.start_indexes, feature.end_indexes, feature.polarity_labels, feature.label_masks):
                if mask and start_index in feature.token_to_orig_map and end_index in feature.token_to_orig_map:
                    final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                        do_lower_case, verbose_logging, logger)
                    if final_text in seen_predictions:
                        continue
                    seen_predictions[final_text] = True

                    span_starts.append(start_index)
                    span_ends.append(end_index)
                    labels.append(polarity_label)
                    label_masks.append(1)
        else:
            prelim_predictions_per_feature = []
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_logit = result.start_logits[start_index]
                    end_logit = result.end_logits[end_index]
                    if start_logit + end_logit < logit_threshold:
                        continue

                    prelim_predictions_per_feature.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=start_logit,
                            end_logit=end_logit))

            if use_heuristics:
                prelim_predictions_per_feature = sorted(
                    prelim_predictions_per_feature,
                    key=lambda x: (x.start_logit + x.end_logit - (x.end_index - x.start_index + 1)),
                    reverse=True)
            else:
                prelim_predictions_per_feature = sorted(
                    prelim_predictions_per_feature,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)

            for i, pred_i in enumerate(prelim_predictions_per_feature):
                if len(span_starts) >= int(n_best_size)/2:
                    break
                final_text = wrapped_get_final_text(example, feature, pred_i.start_index, pred_i.end_index,
                                                    do_lower_case, verbose_logging, logger)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True

                span_starts.append(pred_i.start_index)
                span_ends.append(pred_i.end_index)
                labels.append(0)
                label_masks.append(1)

                # filter out redundant candidates
                if (i+1) < len(prelim_predictions_per_feature) and use_nms:
                    indexes = []
                    for j, pred_j in enumerate(prelim_predictions_per_feature[(i+1):]):
                        filter_text = wrapped_get_final_text(example, feature, pred_j.start_index, pred_j.end_index,
                                                             do_lower_case, verbose_logging, logger)
                        if filter_type == 'em':
                            if exact_match_score(final_text, filter_text):
                                indexes.append(i + j + 1)
                        elif filter_type == 'f1':
                            if f1_score(final_text, filter_text) > 0:
                                indexes.append(i + j + 1)
                        else:
                            raise Exception
                    [prelim_predictions_per_feature.pop(index - k) for k, index in enumerate(indexes)]

        # Pad to fixed length
        while len(span_starts) < int(n_best_size):
            span_starts.append(0)
            span_ends.append(0)
            labels.append(0)
            label_masks.append(0)
        assert len(span_starts) == int(n_best_size)
        assert len(span_ends) == int(n_best_size)
        assert len(labels) == int(n_best_size)
        assert len(label_masks) == int(n_best_size)

        batch_span_starts.append(span_starts)
        batch_span_ends.append(span_ends)
        batch_labels.append(labels)
        batch_label_masks.append(label_masks)
    return batch_span_starts, batch_span_ends, batch_labels, batch_label_masks


def ts2start_end(ts_tag_sequence):
    starts, ends = [], []
    n_tag = len(ts_tag_sequence)
    prev_pos, prev_sentiment = '$$$', '$$$'
    tag_on = False
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag != 'O':
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
        else:
            cur_pos, cur_sentiment = 'O', '$$$'
        assert cur_pos == 'O' or cur_pos == 'T'
        if cur_pos == 'T':
            if prev_pos != 'T':
                # cur tag is at the beginning of the opinion target
                starts.append(i)
                tag_on = True
            else:
                if cur_sentiment != prev_sentiment:
                    # prev sentiment is not equal to current sentiment
                    ends.append(i - 1)
                    starts.append(i)
                    tag_on = True
        else:
            if prev_pos == 'T':
                ends.append(i - 1)
                tag_on = False
        prev_pos = cur_pos
        prev_sentiment = cur_sentiment
    if tag_on:
        ends.append(n_tag-1)
    assert len(starts) == len(ends), (len(starts), len(ends), ts_tag_sequence)
    return starts, ends


def ts2polarity(words, ts_tag_sequence, starts, ends):
    polarities = []
    for start, end in zip(starts, ends):
        cur_ts_tag = ts_tag_sequence[start]
        cur_pos, cur_sentiment = cur_ts_tag.split('-')
        assert cur_pos == 'T'
        prev_sentiment = cur_sentiment
        if start < end:
            for idx in range(start, end + 1):
                cur_ts_tag = ts_tag_sequence[idx]
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                assert cur_pos == 'T'
                assert cur_sentiment == prev_sentiment, (words, ts_tag_sequence, start, end)
                prev_sentiment = cur_sentiment
        polarities.append(cur_sentiment)
    return polarities


def pos2term(words, starts, ends):
    term_texts = []
    for start, end in zip(starts, ends):
        term_texts.append(' '.join(words[start:end+1]))
    return term_texts

def convert_absa_data(dataset, verbose_logging=False):
    examples = []
    n_records = len(dataset)
    for i in range(n_records):
        words = dataset[i]['words']
        terms_tags = dataset[i]['ts_raw_tags']
        starts, ends = ts2start_end(terms_tags)
        polarities = ts2polarity(words, terms_tags, starts, ends)
        term_texts = pos2term(words, starts, ends)

        if term_texts != []:
            new_polarities = []
            for polarity in polarities:
                if polarity == 'POS':
                    new_polarities.append('positive')
                elif polarity == 'NEG':
                    new_polarities.append('negative')
                elif polarity == 'NEU':
                    new_polarities.append('neutral')
                else:
                    raise Exception
            assert len(term_texts) == len(starts)
            assert len(term_texts) == len(new_polarities)
            example = SemEvalExample(str(i), words, term_texts, starts, ends, new_polarities)
            examples.append(example)
            if i < 50 and verbose_logging:
                print(example)
    print("Convert %s examples" % len(examples))
    return examples

def get_aste_examples(dataset, set_type):
    examples = []
    max_len = 0
    for i, instance in enumerate(dataset):
        guid = "%s-%s" % (set_type, i)
        words = instance["words"]
        labels = instance["terms_tags"]
        if len(words) > max_len:
            max_len = len(words)
        examples.append(
            InputExample(guid=guid, text=words, label=labels) )
    print(max_len)
    return examples

def get_aste_cls_examples_withlc(dataset, set_type):
    examples = []
    max_len = 0
    for i, instance in enumerate(dataset):
        guid = "%s-%s" % (set_type, i)
        words = instance["words"]
        aspect = instance['aspect']
        opinion = instance['opinion']
        label = instance['polarity']
        aspect_span = instance['aspect_span']
        opinion_span = instance['opinion_span']
        local_context_span = instance['local_context_span']
        aspect_length = instance ['aspect_length']
        opinion_length = instance ['opinion_length']
        if len(words) > max_len:
            max_len = len(words)
        examples.append(
            InputClsLcExample(guid=guid, text=words, aspect=aspect, opinion=opinion, label=label,
                            aspect_span=aspect_span, opinion_span=opinion_span, local_context_span=local_context_span,
                            aspect_length=aspect_length, opinion_length=opinion_length) )
    print(max_len)
    return examples

def get_aste_cls_examples(dataset, set_type):
    examples = []
    max_len = 0
    for i, instance in enumerate(dataset):
        guid = "%s-%s" % (set_type, i)
        words = instance["words"]
        aspect = instance['aspect']
        opinion = instance['opinion']
        label = instance['polarity']
        if len(words) > max_len:
            max_len = len(words)
        examples.append(
            InputClsExample(guid=guid, text=words, aspect=aspect, opinion=opinion, label=label) )
    print(max_len)
    return examples

def get_aste_pipeline_cls_examples(dataset, set_type):
    examples = []
    max_len = 0
    for i, instance in enumerate(dataset):
        guid = "%s-%s" % (set_type, i)
        example_id = instance['line_id']
        words = instance["words"]
        aspect = instance['aspect']
        opinion = instance['opinion']
        label = instance['polarity']
        gold_triplets = instance['gold_triplets']
        if len(words) > max_len:
            max_len = len(words)
        examples.append(
            InputPipelineClsExample(guid=guid, example_id=example_id, text=words, aspect=aspect, opinion=opinion, label=label, gold_triplets=gold_triplets) )
    print(max_len)
    return examples

def get_aste_pipeline_cls_ls_examples(dataset, set_type):
    examples = []
    max_len = 0
    for i, instance in enumerate(dataset):
        guid = "%s-%s" % (set_type, i)
        example_id = instance['line_id']
        words = instance["words"]
        aspect = instance['aspect']
        opinion = instance['opinion']
        label = instance['polarity']
        gold_triplets = instance['gold_triplets']

        aspect_span = instance['aspect_span']
        opinion_span = instance['opinion_span']
        local_context_span = instance['local_context_span']
        aspect_length = instance['aspect_length']
        opinion_length = instance['opinion_length']
        if len(words) > max_len:
            max_len = len(words)
        examples.append(
            InputPipelineClsLcExample(guid=guid, example_id=example_id, text=words, aspect=aspect, opinion=opinion, label=label, gold_triplets=gold_triplets,
                                    aspect_span=aspect_span, opinion_span=opinion_span,
                                    local_context_span=local_context_span,
                                    aspect_length=aspect_length, opinion_length=opinion_length
                                    ) )
    print(max_len)
    return examples

def span2term(words, span):
    if len(span) > 1:
        term = ' '.join(words[span[0]:span[-1] + 1])
    else:
        term = words[span[0]]
    return term

def get_terms_rel(rel_pairs, words):
    pairs_rel = {}
    # pairs_terms = []
    # pairs_rel = []
    for pair in rel_pairs:
        aspect_span, opinion_span, polarity = pair
        aspect_term = span2term(words, aspect_span)
        opinion_term = span2term(words, opinion_span)
        pairs_rel[aspect_term + '*' + opinion_term] = polarity
        # pairs_terms.append((aspect_term, opinion_term))
        # pairs_rel.append(polarity)
    return pairs_rel

def read_aste_cls_data(train_path, pair_path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    fr = open(pair_path, 'rb')
    all_relation_pairs = pickle.load(fr)
    dataset = []
    polarity_list = []
    len_all_pairs, len_pos, len_neg, len_neu = 0,0,0,0
    with open(train_path, encoding='UTF-8') as fp:
        for index,line in enumerate(fp):
            words, aspect_tags = process_aspect(line)
            opinion_tags = process_opinion(line)
            # 得到所有的aspect，opinion的集合，然后n*m循环，若pair在rel_pairs里面则情感为对应的，否则情感为NONE
            aspect_pos = get_tags_BIOES(aspect_tags)
            opinion_pos = get_tags_BIOES(opinion_tags)
            aspect_terms = get_terms_by_index(words, aspect_pos) # aspect列表
            opinion_terms = get_terms_by_index(words, opinion_pos) # opinion列表

            #aspect和opinion去重
            aspect_terms = list(set(aspect_terms))
            opinion_terms = list(set(opinion_terms))

            # 所有关系三元组
            rel_pairs = all_relation_pairs[index]  # [(aspectpos[beg,end], opinionpos[beg,end], senti),(),()]
            pairs_rel = get_terms_rel(rel_pairs, words) # [(aspectterm, opinionterm)] [senti]
            len_all_pairs += len(pairs_rel)
            # for debug
            # if len(pairs_rel) != len(rel_pairs):
            #     print("hi")
            # for k in pairs_rel:
            #     if pairs_rel[k] == 0: len_neg += 1
            #     elif pairs_rel[k] == 1: len_pos += 1
            #     elif pairs_rel[k] == 2: len_neu += 1
            #     else: print(pairs_rel[k])
            #     aspect = k.split('*')[0]
            #     opinion = k.split('*')[1]
            #     if aspect not in aspect_terms:
            #         print(aspect)
            #     if opinion not in opinion_terms:
            #         print(opinion)

            for aspect in aspect_terms:
                for opinion in opinion_terms:
                    records = {}
                    records['line_raw'] = line
                    records['words'] = words
                    records['aspect'] = aspect
                    records['opinion'] = opinion
                    ##################################加一个aspect和opinion之间的位置，取对应位置标示 & 距离特征
                    pair_term = aspect + '*' +opinion
                    if pair_term in pairs_rel.keys():
                        records['polarity'] = pairs_rel[pair_term]
                    else:
                        records['polarity'] = sent2id['NONE']  # 3
                    polarity_list.append(records['polarity'])
                    dataset.append(records)
    len_examples = len(dataset)
    len_pos_exp = polarity_list.count(1)
    len_neg_exp = polarity_list.count(0)
    len_neu_exp = polarity_list.count(2)
    len_none_exp = polarity_list.count(3)
    print("Obtain %s records from %s" % (len(dataset), train_path))
    return dataset

def read_aste_cls_data_withlc(train_path, pair_path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    fr = open(pair_path, 'rb')
    all_relation_pairs = pickle.load(fr)
    dataset = []
    polarity_list = []
    len_all_pairs, len_pos, len_neg, len_neu = 0,0,0,0
    with open(train_path, encoding='UTF-8') as fp:
        for index,line in enumerate(fp):
            words, aspect_tags = process_aspect(line)
            opinion_tags = process_opinion(line)
            # 得到所有的aspect，opinion的集合，然后n*m循环，若pair在rel_pairs里面则情感为对应的，否则情感为NONE
            aspect_pos = get_tags_BIOES(aspect_tags)    # beg, end, type
            opinion_pos = get_tags_BIOES(opinion_tags)
            aspect_terms = get_terms_by_index(words, aspect_pos) # aspect列表
            opinion_terms = get_terms_by_index(words, opinion_pos) # opinion列表

            #aspect和opinion去重
            # aspect_terms = list(set(aspect_terms))
            # opinion_terms = list(set(opinion_terms))

            # 位置与term映射
            aspect_dic = {}
            opinion_dic = {}
            for pos, at in zip(aspect_pos, aspect_terms):
                beg, end, type = pos
                aspect_dic[(beg, end)] = at
            for pos, ot in zip(opinion_pos, opinion_terms):
                beg, end, type = pos
                opinion_dic[(beg, end)] = ot


            # 所有关系三元组
            rel_pairs = all_relation_pairs[index]  # [(aspectpos[beg,end], opinionpos[beg,end], senti),(),()]
            pairs_rel = get_terms_rel(rel_pairs, words) # [(aspectterm, opinionterm)] [senti]
            len_all_pairs += len(pairs_rel)

            for aspect_span in aspect_dic.keys():
                for opinion_span in opinion_dic.keys():
                    aspect = aspect_dic[aspect_span]
                    opinion = opinion_dic[opinion_span]
                    records = {}
                    records['line_raw'] = line
                    records['words'] = words
                    records['aspect'] = aspect
                    records['opinion'] = opinion
                    records['aspect_span'] = aspect_span
                    records['opinion_span'] = opinion_span
                    records['local_context_span'] = (min(aspect_span[1],opinion_span[1]), max(aspect_span[0], opinion_span[0]) )
                    records['aspect_length'] = len(aspect.split())
                    records['opinion_length'] = len(opinion.split())

                    # 根据terms来算对，还是必须pos也对？
                    pair_term = aspect + '*' + opinion
                    if pair_term in pairs_rel.keys():
                        records['polarity'] = pairs_rel[pair_term]
                    else:
                        records['polarity'] = sent2id['NONE']  # 3
                    polarity_list.append(records['polarity'])
                    dataset.append(records)

    print("Obtain %s records from %s" % (len(dataset), train_path))
    return dataset

def read_aste_pipeline_cls_data(predict_path, pair_path, terms_extraction_path):
    fr = open(pair_path, 'rb')
    all_relation_pairs = pickle.load(fr)  # gold
    with open(terms_extraction_path) as ft:
        terms_extraction = json.load(ft)  # extract aspect、opinion pair
    dataset = []
    with open(predict_path, encoding='UTF-8') as fp:
        for index,line in enumerate(fp):
            words = process_words(line)
            # 得到predict的所有的aspect，opinion的集合，然后n*m循环，若pair在rel_pairs里面则情感为对应的，否则情感为NONE
            terms_extract = terms_extraction[str(index)]
            aspect_terms = terms_extract['pred_aspect']
            opinion_terms = terms_extract['pred_opinion']

            aspect_terms = list(set(aspect_terms))
            opinion_terms = list(set(opinion_terms))

            # 所有gold关系三元组，存下来便于统计F R P
            rel_pairs = all_relation_pairs[index]  # [(aspectpos[beg,end], opinionpos[beg,end], senti),(),()]
            pairs_rel = get_terms_rel(rel_pairs, words) # {aspectterm-opinionterm : senti}
            assert len(rel_pairs) == len(pairs_rel)

            for aspect in aspect_terms:
                for opinion in opinion_terms:
                    records = {}
                    records['line_id'] = index
                    records['line_raw'] = line
                    records['words'] = words
                    records['aspect'] = aspect
                    records['opinion'] = opinion
                    records['gold_triplets'] = pairs_rel
                    pair_term = aspect + '*' +opinion
                    if pair_term in pairs_rel.keys():
                        records['polarity'] = pairs_rel[pair_term]
                    else:
                        records['polarity'] = sent2id['NONE']  # 3
                    dataset.append(records)

    print("Obtain %s records from %s" % (len(dataset), terms_extraction_path))
    return dataset

def read_aste_pipeline_cls_lc_data(predict_path, pair_path, terms_extraction_path):
    fr = open(pair_path, 'rb')
    all_relation_pairs = pickle.load(fr)  # gold
    with open(terms_extraction_path) as ft:
        terms_extraction = json.load(ft)  # extract aspect、opinion pair
    dataset = []
    with open(predict_path, encoding='UTF-8') as fp:
        for index,line in enumerate(fp):
            words = process_words(line)
            # 得到predict的所有的aspect，opinion的集合，然后n*m循环，若pair在rel_pairs里面则情感为对应的，否则情感为NONE
            ts = terms_extraction[index]

            # 所有gold关系三元组，存下来便于统计F R P
            rel_pairs = all_relation_pairs[index]  # [(aspectpos[beg,end], opinionpos[beg,end], senti),(),()]
            pairs_rel = get_terms_rel(rel_pairs, words)  # {aspectterm-opinionterm : senti}
            assert len(rel_pairs) == len(pairs_rel)

            words_2 = ts['tokens']
            assert words == words_2
            terms = ts['entities']
            at_list = []
            ot_list = []
            for id, term_dic in enumerate(terms):
                if term_dic['type'] == 'aspect':
                    aspect_dic = {}
                    aspect_dic['id'] = id
                    aspect_dic['span'] = (term_dic['start'], term_dic['end']-1)
                    aspect_dic['term'] = words[term_dic['start']: term_dic['end']]
                    at_list.append(aspect_dic)
                elif term_dic['type'] == 'opinion':
                    opinion_dic = {}
                    opinion_dic['id'] = id
                    opinion_dic['span'] = (term_dic['start'], term_dic['end']-1)
                    opinion_dic['term'] = words[term_dic['start']: term_dic['end']]
                    ot_list.append(opinion_dic)

            for at in at_list:
                for ot in ot_list:
                    records = {}
                    records['line_id'] = index
                    records['line_raw'] = line
                    records['words'] = words
                    records['aspect'] = ' '.join(at['term'])
                    records['opinion'] = ' '.join(ot['term'])
                    records['aspect_span'] = at['span']
                    records['opinion_span'] = ot['span']
                    records['local_context_span'] = (min(at['span'][1], ot['span'][1]), max(at['span'][0], ot['span'][0]))
                    records['aspect_length'] = len(at['term'])
                    records['opinion_length'] = len(ot['term'])
                    records['gold_triplets'] = pairs_rel

                    pair_term = records['aspect']  + '*' + records['opinion']
                    if pair_term in pairs_rel.keys():
                        records['polarity'] = pairs_rel[pair_term]
                    else:
                        records['polarity'] = sent2id['NONE']  # 3
                    dataset.append(records)

    print("Obtain %s records from %s" % (len(dataset), terms_extraction_path))
    return dataset

def get_tags_BIOES(ts_tag_sequence):
    """
    transform ts tag sequence to targeted sentiment
    :param ts_tag_sequence: tag sequence for ts task
    :return:
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence,types = [],[]
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        # current position and sentiment
        # tag O and tag EQ will not be counted
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, type = eles
        else:
            pos, type = 'O', 'O'
        if type != 'O':
            # current word is a subjective word
            types.append(type)
        if pos == 'S':
            # singleton
            ts_sequence.append((i, i, type))
            types = []
        elif pos == 'B':
            beg = i
            if len(types) > 1:
                # remove the effect of the noisy I-{POS,NEG,NEU}
                types = [types[-1]]
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(types)) == 1:
                ts_sequence.append((beg, end, type))
                types = []
                beg, end = -1, -1

    return ts_sequence

def get_terms_by_index(words, pre_tags):
    terms = []
    for tag in pre_tags:
        beg, end, type = tag
        terms.append(' '.join(words[beg:end+1])) # CLS
    return terms

def read_absa_data_lap(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string, _ = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            terms_tags = []
            # tag sequence for opinion target extraction
            ote_tags = []
            # word sequence
            words = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                words.append(word.lower())
                if tag == 'O':
                    ote_tags.append('O')
                    terms_tags.append('O')
                elif tag.split('-')[1] == 'POS':
                    ote_tags.append('T')
                    terms_tags.append('T-POS')
                elif tag.split('-')[1] == 'NEG':
                    ote_tags.append('T')
                    terms_tags.append('T-NEG')
                elif tag.split('-')[1] == 'NEU':
                    ote_tags.append('T')
                    terms_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['ote_raw_tags'] = ote_tags.copy()
            record['ts_raw_tags'] = terms_tags.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset

def read_aste_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            words, aspect_tags = process_aspect(line)
            opinion_tags = process_opinion(line)
            # 通过aspect和opinion的tag整合起来，得到terms_tags，判断每个词是不是只有一个tag，有没有嵌套情况
            assert len(aspect_tags) == len(opinion_tags)
            terms_tags = []
            for i,word_tag in enumerate(aspect_tags):
                if opinion_tags[i] != 'O':
                    terms_tags.append(opinion_tags[i])
                    if word_tag != 'O':
                        print("嵌套????")
                else:
                    terms_tags.append(word_tag)
            records = {}
            records["line_raw"] = line
            records["words"] = words
            records["aspect_tags"] = aspect_tags
            records["opinion_tags"] = opinion_tags
            records["terms_tags"] = terms_tags
            dataset.append(records)

            print("Obtain %s records from %s" % (len(dataset), path))
    return dataset

def process_words(line):
    words_tag = line.strip().split("####")
    words = words_tag[0].strip().split()
    words = [w.lower() for w in words]
    return words

def process_aspect(line):
    words = []
    terms_tags = []

    words_tag = line.strip().split("####")
    prev = None
    for i,p in enumerate(words_tag[1].split(" ")):
        word, tag = p.split("=")
        words.append(word.lower())

        t = tag
        if t == "O":
            if prev != None:
                if terms_tags[-1]== "B-aspect":
                    terms_tags[-1] = "S-aspect"
                else:
                    terms_tags[-1] = "E-aspect"
                prev=None

            terms_tags.append("O")

        else:
            t_p = t
            if prev == None:
                prev = t_p
                terms_tags.append("B-aspect")
            else:
                if prev!= t_p:
                    if terms_tags[-1]=="B-aspect":
                        terms_tags[-1] = "S-aspect"
                    else:
                        terms_tags[-1] = "E-aspect"
                    # prev = None

                    prev = t_p
                    terms_tags.append("B-aspect")
                else:
                    terms_tags.append("I-aspect")

    if prev!=None:
        if terms_tags[-1]=="B-aspect":
            terms_tags[-1] = "S-aspect"

        else:
            terms_tags[-1] = "E-aspect"
    assert len(words)==len(terms_tags)

    return words,terms_tags


def process_opinion(line):
    opinion_tags = []
    words_tag = line.strip().split("####")
    word_opinion = line.strip().split("####")[2]
    # if line.strip()=="I can barely use any usb devices because they will not stay connected properly .####I=O can=O barely=O use=O any=O usb=T-NEG devices=T-NEG because=O they=O will=O not=O stay=O connected=O properly=O .=O####I=O can=O barely=O use=O any=O usb=O devices=O because=O they=O will=O not=S stay=S connected=S properly=S .=O":
    #     print("hi")

    prev = None
    for i,p in enumerate(word_opinion.split()):
        word, tag = p.split("=")
        if tag != "O":
            if tag==prev:
                opinion_tags.append("I-opinion")
            else:
                opinion_tags.append("B-opinion")
            prev = tag
        else:
            if prev != None:
                if opinion_tags[-1] == "B-opinion":
                    opinion_tags[-1] = "S-opinion"
                elif opinion_tags[-1] == "I-opinion":
                    opinion_tags[-1] = "E-opinion"
            opinion_tags.append('O')
            prev = "O"

    if prev!=None:
        if opinion_tags[-1] == "B-opinion":
            opinion_tags[-1] = "S-opinion"
        elif opinion_tags[-1] == "I-opinion":
            opinion_tags[-1] = "E-opinion"

    assert len(opinion_tags) == len(words_tag[0].split(" "))
    return opinion_tags
