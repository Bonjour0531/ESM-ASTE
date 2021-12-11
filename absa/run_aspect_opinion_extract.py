"""Run BERT on SemEval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import pickle
import argparse
import collections

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bert.tokenization as tokenization
from bert.modeling import BertConfig
from bert.sentiment_modeling import BertForBIOAspectAndOpinionExtraction

from squad.squad_evaluate import exact_match_score, metric_max_over_ground_truths
from absa.utils import read_aste_data, get_aste_examples,convert_absa_data, convert_examples_to_features, RawFinalResult, RawSpanResult, \
    span_annotate_candidates, wrapped_get_final_text
from absa.run_base import copy_optimizer_params_to_model, set_optimizer_params_grad, prepare_optimizer, post_process_loss, bert_load_state_dict

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def read_train_data(args, tokenizer, logger):
    train_path = os.path.join(args.data_dir, args.train_file)
    train_set = read_aste_data(train_path)
    train_examples = get_aste_examples(dataset=train_set, set_type="train")

    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length,logger)

    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Batch size = %d", args.train_batch_size)
    logger.info("Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader, num_train_steps

def read_eval_data(args, tokenizer, logger, state):
    if state == "eval":
        eval_path = os.path.join(args.data_dir, args.eval_file)
    elif state == "predict":
        eval_path = os.path.join(args.data_dir, args.predict_file)
    eval_set = read_aste_data(eval_path)
    eval_examples = get_aste_examples(dataset=eval_set, set_type="eval")

    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,logger)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Batch size = %d", args.predict_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, eval_dataloader

def run_train_epoch(args, epoch, global_step, model, param_optimizer, train_dataloader,
                    eval_examples, eval_features, eval_dataloader,
                    optimizer, n_gpu, device, logger, log_path, save_path,
                    save_checkpoints_steps, start_save_steps, best_f1):
    running_loss, count = 0.0, 0
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        loss = post_process_loss(args, n_gpu, loss)
        loss.backward()
        running_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 or args.optimize_on_cpu:
                if args.fp16 and args.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        param.grad.data = param.grad.data / args.loss_scale
                is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    args.loss_scale = args.loss_scale / 2
                    model.zero_grad()
                    continue
                optimizer.step()
                copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
            else:
                optimizer.step()
            model.zero_grad()
            global_step += 1
            count += 1

            if global_step % save_checkpoints_steps == 0 and count != 0:
                logger.info("step: {}, loss: {:.4f}".format(global_step, running_loss / count))

            if global_step % save_checkpoints_steps == 0 and global_step > start_save_steps and count != 0:  # eval & save esm
                logger.info("***** Running evaluation *****")
                model.eval()
                metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger)
                f = open(log_path, "a")
                print("step: {}, loss: {:.4f}, macro-F1: {:.4f}, micro-F1: {:.4f},micro-P: {:.4f}, micro-R: {:.4f}, aspect-F1: {:.4f},aspect-P: {:.4f}, aspect-R: {:.4f}, opinion-F1: {:.4f},opinion-P: {:.4f}, opinion-R: {:.4f}"
                      .format(global_step,running_loss / count,
                              metrics['macro-f1'], metrics['micro-f1'], metrics['micro-precision'], metrics['micro-recall'],
                              metrics['aspect-f1'], metrics['aspect-p'], metrics['aspect-r'],
                              metrics['opinion-f1'],metrics['opinion-p'],metrics['opinion-r']), file=f)
                print(" ", file=f)
                f.close()
                running_loss, count = 0.0, 0
                model.train()
                if metrics['macro-f1'] > best_f1: ########################################################用R试试
                    best_f1 = metrics['macro-f1']
                    torch.save({
                        'esm': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': global_step
                    }, save_path)
                    f = open(log_path, "a")
                    print(f"best_f1 in epoch:{epoch + 1} ", file=f)
                    print(" ", file=f)
                    f.close()
                if args.debug:
                    break
    return global_step, model, best_f1

def eval_aspect_extract(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        for start_index, end_index, span_mask in zip(result.start_indexes, result.end_indexes, result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)

        prediction = {'pred': pred_terms, 'gold': example.term_texts}
        all_nbest_json[example.example_id] = prediction

        for pred_term in pred_terms:
            common += metric_max_over_ground_truths(exact_match_score, pred_term, example.term_texts)
        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.

    return {'p': p, 'r': r, 'f1': f1, 'common': common, 'retrieved': retrieved, 'relevant': relevant}, all_nbest_json

def compute_metrics_aeoe(eval_examples, outputs,labels):
    tag2id = {'O': 0, 'B-aspect': 1, 'I-aspect': 2, 'E-aspect': 3, 'S-aspect': 4,
                 'B-opinion': 5, 'I-opinion': 6, 'E-opinion': 7, 'S-opinion': 8}
    id2tag =  {v: k for k, v in tag2id.items()}
    aspect_origin, opinion_origin = 0., 0.
    aspect_found, opinion_found = 0., 0.
    aspect_right, opinion_right = 0., 0.
    all_predict_json = collections.OrderedDict()
    for index, fetch in enumerate(zip(eval_examples,labels, outputs)):
        example, gold, pre = fetch
        pred_tags = [id2tag[label] for label in pre]
        gold_tags = [id2tag[label] for label in gold]
        gold_aspect_tags, gold_opinion_tags = get_tags_BIOES(gold_tags)
        pre_aspect_tags, pre_opinion_tags = get_tags_BIOES(pred_tags)

        gold_aspect_terms = get_terms_by_index(example, gold_aspect_tags)
        gold_opinion_terms = get_terms_by_index(example, gold_opinion_tags)
        pre_aspect_terms = get_terms_by_index(example, pre_aspect_tags)
        pre_opinion_terms = get_terms_by_index(example, pre_opinion_tags)

        # predict 的aspect和opinion去重
        # pre_aspect_terms = list(set(pre_aspect_terms))
        # pre_opinion_terms = list(set(pre_opinion_terms))

        prediction = {'pred_aspect': pre_aspect_terms, 'gold_aspect': gold_aspect_terms,
                      'pred_opinion': pre_opinion_terms, 'gold_opinion':gold_opinion_terms}
        all_predict_json[example.guid.split('-')[1]] = prediction

        aspect_origin += len(gold_aspect_tags)
        opinion_origin += len(gold_opinion_tags)
        aspect_found += len(pre_aspect_tags)
        opinion_found += len(pre_opinion_tags)

        for p_tag in pre_aspect_tags:
            if p_tag in gold_aspect_tags:
                aspect_right += 1
        for p_tag in pre_opinion_tags:
            if p_tag in gold_opinion_tags:
                opinion_right += 1

        # aspect_origin += len(gold_aspect_terms)
        # opinion_origin += len(gold_opinion_terms)
        # aspect_found += len(pre_aspect_terms)
        # opinion_found += len(pre_opinion_terms)
        #
        # for p_term in pre_aspect_terms:
        #     if p_term in gold_aspect_terms:
        #         aspect_right += 1
        # for p_term in gold_opinion_terms:
        #     if p_term in pre_opinion_terms:
        #         opinion_right += 1
    aspect_precision, aspect_recall, aspect_f1 = get_recall_f1(aspect_origin, aspect_found, aspect_right)
    opinion_precision, opinion_recall, opinion_f1 = get_recall_f1(opinion_origin, opinion_found, opinion_right)
    macro_f1 = (aspect_f1 + opinion_f1) / 2
    micro_precision, micro_recall, micro_f1 = get_recall_f1(aspect_origin + opinion_origin, aspect_found+opinion_found, aspect_right+opinion_right)
    metrics = {'macro-f1': macro_f1, "micro-f1": micro_f1, 'micro-precision': micro_precision, "micro-recall": micro_recall,
               'aspect-f1':aspect_f1,'aspect-p':aspect_precision,'aspect-r':aspect_recall,
               'opinion-f1':opinion_f1,'opinion-p':opinion_precision,'opinion-r':opinion_recall,
               }

    return metrics, all_predict_json

def get_terms_by_index(example, pre_tags):
    words = example.text
    terms = []
    for tag in pre_tags:
        beg, end, type = tag
        terms.append(' '.join(words[beg-1:end])) # CLS
    return terms


def get_recall_f1(origin, found, right):
    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

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
    aspect_terms = []
    opinion_terms = []
    for term in ts_sequence:
        beg, end, type = term
        if type == "aspect":
            aspect_terms.append(term)
        else:
            opinion_terms.append(term)
    return aspect_terms, opinion_terms


def evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False, do_pipeline=False):
    all_results = []
    predicts_all = []
    labels_all = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, example_indices = batch
        with torch.no_grad():
            predictions = model(input_ids, segment_ids, input_mask)

        predictions = predictions.detach().cpu().numpy().tolist()
        label_ids = label_ids.to('cpu').numpy().tolist()

        predicts_all.extend(predictions)
        labels_all.extend(label_ids)
    # 获取评价指标与预测结果
    metrics, all_predict_json = compute_metrics_aeoe(eval_examples, predicts_all, labels_all)

    if write_pred:
        # 存储预测结果
        output_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_file, "w") as writer:
            writer.write(json.dumps(all_predict_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_file))

    # do_pipeline这里看情况修改，直接保存结果，words, aspect,opinion
    if do_pipeline:
        output_file = os.path.join(args.output_dir, "extraction_results.pkl")
        pickle.dump(all_results, open(output_file, 'wb'))
    return metrics


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_config_file", default='bert-base-uncased/config.json', type=str,
                        help="The config json file corresponding to the pre-trained BERT esm. "
                             "This specifies the esm architecture.")
    parser.add_argument("--vocab_file", default='bert-base-uncased/vocab.txt', type=str,
                        help="The vocabulary file that the BERT esm was trained on.")
    parser.add_argument("--output_dir", default='../out/extract/14lap_ep10', type=str,
                        help="The output directory where the esm checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--data_dir", default='../data/SemEval-Triplet-data-master/triplet_data_only/14lap', type=str, help="SemEval data dir")
    parser.add_argument("--train_file", default='train.txt', type=str, help="SemEval xml for training")
    parser.add_argument("--eval_file", default='dev.txt', type=str, help="SemEval xml for training")
    parser.add_argument("--predict_file", default='test.txt', type=str, help="SemEval csv for prediction")
    parser.add_argument("--init_checkpoint", default='bert-base-uncased/pytorch_model.bin', type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT esm).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=96, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pipeline", default=False, action='store_true', help="Whether to run pipeline on the dev set.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_proportion", default=0.5, type=float,
                        help="Proportion of steps to save models for. E.g., 0.5 = 50% of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=12, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--logit_threshold", default=7.5, type=float,
                        help="Logit threshold for annotating labels.")
    parser.add_argument("--filter_type", default="f1", type=str, help="Which filter type to use")
    parser.add_argument("--use_heuristics", default=True, action='store_true',
                        help="If true, use heuristic regularization on span length")
    parser.add_argument("--use_nms", default=True, action='store_true',
                        help="If true, use nms to prune redundant spans")
    parser.add_argument("--verbose_logging", default=True, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    if not args.do_train and not args.do_predict and not args.do_pipeline:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train and not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict and not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("torch_version: {} device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        torch.__version__, device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT esm "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info('output_dir: {}'.format(args.output_dir))
    save_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    log_path = os.path.join(args.output_dir, 'performance.txt')
    network_path = os.path.join(args.output_dir, 'network.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter.txt')

    f = open(parameter_path, "w")
    for arg in sorted(vars(args)):
        print("{}: {}".format(arg, getattr(args, arg)), file=f)
    f.close()

    logger.info("***** Preparing esm *****")
    model = BertForBIOAspectAndOpinionExtraction(bert_config)
    if args.init_checkpoint is not None and not os.path.isfile(save_path):
        model = bert_load_state_dict(model, torch.load(args.init_checkpoint, map_location='cpu'))
        logger.info("Loading esm from pretrained checkpoint: {}".format(args.init_checkpoint))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['esm'])
        step = checkpoint['step']
        logger.info("Loading esm from finetuned checkpoint: '{}' (step {})"
                    .format(save_path, step))

    f = open(network_path, "w")
    for n, param in model.named_parameters():
        print("name: {}, size: {}, dtype: {}, requires_grad: {}"
              .format(n, param.size(), param.dtype, param.requires_grad), file=f)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: {}".format(total_trainable_params), file=f)
    print("Total parameters: {}".format(total_params), file=f)
    f.close()

    logger.info("***** Preparing data *****")
    train_dataloader, num_train_steps = None, None
    eval_examples, eval_features, eval_dataloader = None, None, None
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    if args.do_train:
        logger.info("***** Preparing training *****")
        train_dataloader, num_train_steps = read_train_data(args, tokenizer, logger)
        logger.info("***** Preparing evaluation *****")
        eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger, state="eval")

    logger.info("***** Preparing optimizer *****")
    optimizer, param_optimizer = prepare_optimizer(args, model, num_train_steps)

    global_step = 0
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        logger.info("Loading optimizer from finetuned checkpoint: '{}' (step {})".format(save_path, step))
        global_step = step

    if args.do_train:
        logger.info("***** Running training *****")
        best_f1 = 0
        save_checkpoints_steps = int(num_train_steps / (5 * args.num_train_epochs))
        start_save_steps = int(num_train_steps * args.save_proportion)
        if args.debug:
            args.num_train_epochs = 1
            save_checkpoints_steps = 20
            start_save_steps = 0
        model.train()
        for epoch in range(int(args.num_train_epochs)):
            logger.info("***** Epoch: {} *****".format(epoch+1))
            global_step, model, best_f1 = run_train_epoch(args, epoch, global_step, model, param_optimizer,
                                                           train_dataloader, eval_examples, eval_features, eval_dataloader,
                                                           optimizer, n_gpu, device, logger, log_path, save_path,
                                                           save_checkpoints_steps, start_save_steps, best_f1)

    if args.do_predict:
        logger.info("***** Running prediction *****")
        eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger, state='predict')

        # restore from best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['esm'])
            step = checkpoint['step']
            logger.info("Loading esm from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))

        model.eval()
        metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=True)
        f = open(log_path, "a")
        print("threshold: {}, step: {}, macro-F1: {:.4f}, micro-F1: {:.4f},micro-P: {:.4f}, micro-R: {:.4f}, aspect-F1: {:.4f},aspect-P: {:.4f}, aspect-R: {:.4f}, opinion-F1: {:.4f},opinion-P: {:.4f}, opinion-R: {:.4f}"
              .format(args.logit_threshold, global_step,
                      metrics['macro-f1'], metrics['micro-f1'], metrics['micro-precision'], metrics['micro-recall'],
                      metrics['aspect-f1'], metrics['aspect-p'], metrics['aspect-r'],
                      metrics['opinion-f1'], metrics['opinion-p'], metrics['opinion-r']), file=f)
        print(" ", file=f)
        f.close()

    if args.do_pipeline:
        logger.info("***** Running pipeline *****")
        if eval_dataloader is None:
            eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)

        # restore from best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['esm'])
            step = checkpoint['step']
            logger.info("Loading esm from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))

        model.eval()
        metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=True,
                           do_pipeline=True)
        print("step: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {})"
              .format(global_step, metrics['p'], metrics['r'],
                      metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant']))


if __name__=='__main__':
    main()











