"""Evaluate the adversarial connections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import argparse
import random
import json
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling_orig import BertForSequenceClassification, BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from examples.classifier_processer_for_ad import InputExample, InputFeatures, DataProcessor, MrpcProcessor, MnliProcessor, RteProcessor, ScitailProcessor, ColaProcessor, SstProcessor, QqpProcessor, QnliProcessor, WnliProcessor, StsProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "rte": RteProcessor,
    "sst-2": SstProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "wnli": WnliProcessor,
    "sts-b": StsProcessor,
    "scitail": ScitailProcessor,
}

num_labels_task = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "rte": 2,
    "sst-2": 2,
    "qqp": 2,
    "qnli": 2,
    "wnli": 2,
    "sts-b": 1,
    "scitail": 2,
}

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, pattern, if_all):
    """Loads a data file into a list of `InputBatch`s."""
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = None
    
    max_len_count = 0

    features = []
    tokenslist = []
    if pattern["max_combined_attr"][1] / pattern["max_combined_attr"][0] < 0.2 and not if_all:
        target_position = sorted(list(set(pattern["top1position"])))
    else:
        target_position = sorted(list(set(pattern["top1position"] + pattern["top2position"])))
    clean_position = []
    added_tokens = []
    for i in range(len(target_position)):
        if pattern["tokens"][target_position[i]] != "[CLS]" and pattern["tokens"][target_position[i]] != "[SEP]":
            clean_position.append(target_position[i])
            added_tokens.append(pattern["tokens"][target_position[i]])
    target_position = clean_position
    position_type = []    # The adversarial tokens belong to tokens_a or tokens_b

    for i in range(len(target_position)):
        if target_position[i] < pattern["seg_pos"]:   # in tokens_a
            position_type.append(0)
            target_position[i] = target_position[i] / (pattern["seg_pos"] - 1)
        elif target_position[i] > pattern["seg_pos"]:   # in tokens_b
            position_type.append(1)
            target_position[i] = (target_position[i]-pattern["seg_pos"]) / (len(pattern["tokens"]) - 2 - pattern["seg_pos"]) 
        else:
            print("tar_postion should not be seg_pos !")
            exit(0)

    w_token_a_count = position_type.count(0)   # the number of words which should be added in tokens_a
    w_token_b_count = position_type.count(1)   # the number of words which should be added in tokens_b

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        origin_len_a = len(tokens_a)

        prev_pos = -1
        for i in range(w_token_a_count):
            cur_pos = round((origin_len_a+w_token_a_count)*target_position[i])
            if prev_pos == cur_pos:
                cur_pos += 1
            prev_pos = cur_pos
            tokens_a.insert(cur_pos, added_tokens[i])

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            origin_len_b = len(tokens_b)

            prev_pos = -1
            for i in range(w_token_a_count, len(position_type)):
                cur_pos = round((origin_len_b+w_token_b_count)*target_position[i])
                if prev_pos == cur_pos:
                    cur_pos += 1
                prev_pos = cur_pos
                tokens_b.insert(cur_pos, added_tokens[i])

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        base_tokens = ["[UNK]"] + ["[UNK]"]*len(tokens_a) + ["[UNK]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            base_tokens += ["[UNK]"]*len(tokens_b) + ["[UNK]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        if len(tokens) == 128:
            max_len_count += 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        baseline_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(baseline_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if label_map:
            label_id = label_map[example.label]
        else:
            label_id = float(example.label)
        if ex_index < 2:
            logger.debug("*** Example ***")
            logger.debug("guid: %s" % (example.guid))
            logger.debug("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.debug("input_ids: %s" %
                         " ".join([str(x) for x in input_ids]))
            logger.debug("input_mask: %s" %
                         " ".join([str(x) for x in input_mask]))
            logger.debug(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.debug("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          tokens=tokens,
                          baseline_ids=baseline_ids))
        tokenslist.append(tokens)
    print("max_length_count is " + str(max_len_count) + "\n")
    return features, tokenslist


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the experimental results will be written.")
    parser.add_argument("--pattern_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The file which contains the adversarial patterns.")
    parser.add_argument("--model_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The model file which will be evaluated.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # pruning head parameters
    parser.add_argument("--eval_batch_size",
                        default=400,
                        type=int)
    parser.add_argument("--start_exp",
                        default=0,
                        type=int,
                        help="The start index of training examples.")
    parser.add_argument("--num_exp",
                        default=500,
                        type=int,
                        help="The number of training examples for finding patterns.")
    parser.add_argument("--if_all",
                        default=False,
                        action='store_true',
                        help="If peturb the input with all two patterns(top1 and top2), or the threshold is 0.2.")
    parser.add_argument("--data_type",
                        default="train",
                        type=str,
                        help="Patterns from dev_set or training_set.")
    parser.add_argument("--pattern_threshold",
                        default=0.02,
                        type=float,
                        help="The patterns which are less than the threshold will be removed.")


    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))
    assert args.pattern_file.find(args.data_type) != -1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.task_name == 'sts-b':
        lbl_type = torch.float
    else:
        lbl_type = torch.long

    # Load a fine-tuned model 
    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare the data
    eval_set_list = []
    segment_name = "dev_matched" if args.task_name == 'mnli' else "dev" 
    eval_examples = processor.get_dev_examples(args.data_dir, segment=segment_name)

    for i, sub_examples in enumerate(eval_examples):
        eval_set_list.append((label_list[i], sub_examples))
    model.eval()

    # pattern list
    pattern_list = []
    with open(args.pattern_file) as fin:
        pattern_list = json.load(fin)
    
    cur_index = 0
    for cur_index in range(0, len(pattern_list)):
        # the threshold of max_combined_attr
        if sum(pattern_list[cur_index]["max_combined_attr"]) < args.pattern_threshold:
            break
    pattern_list = pattern_list[0:cur_index]
    logger.info("The number of valid patterns is {0} ".format(len(pattern_list)))

    if args.bert_model.find("base") != -1:
        num_head, num_layer = 12, 12
    elif args.bert_model.find("large") != -1:
        num_head, num_layer = 16, 24

    eval_loss, eval_result = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits, all_label_ids = [], []

    # traverse the pattern list and do the evaluation
    for p_index in range(len(pattern_list)):
        seg_result = []
        for eval_segment, eval_examples in eval_set_list:
            eval_features, tokens_list = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, pattern_list[p_index], args.if_all)
            logger.info("***** Running evaluation: %s *****", eval_segment)
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor(
                [f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor(
                [f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor(
                [f.label_id for f in eval_features], dtype=lbl_type)
            eval_data = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_result = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            all_logits, all_label_ids = [], []
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(
                        input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                all_logits.append(logits)
                all_label_ids.append(label_ids)

                eval_loss += tmp_eval_loss.mean().item()

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps

            # compute evaluation metric
            all_logits = np.concatenate(all_logits, axis=0)
            all_label_ids = np.concatenate(all_label_ids, axis=0)
            metric_func = processor.get_metric_func()
            eval_result = metric_func(all_logits, all_label_ids)

            result = {'eval_loss': eval_loss,
                        'eval_result': eval_result,
                        'task_name': args.task_name,
                        'eval_segment': eval_segment}
            seg_result.append((eval_segment, result))
            # logging the results
            logger.info("***** Eval results ({0}: {1}) *****".format(eval_segment, eval_result))

        for eval_segment, result in seg_result:
            pattern_list[p_index][eval_segment] = result["eval_result"]
    
    if not args.if_all:
        out_file_name = "eval_{0}_pattern_one_exp{1}-{2}.json".format(args.data_type, args.start_exp, args.start_exp+args.num_exp)
    else:
        out_file_name = "eval_{0}_pattern_two_exp{1}-{2}.json".format(args.data_type, args.start_exp, args.start_exp+args.num_exp)
    with open(os.path.join(args.output_dir, out_file_name), "w") as fout:
        fout.write(json.dumps(pattern_list, indent=2) + "\n")
    

if __name__ == "__main__":
    main()