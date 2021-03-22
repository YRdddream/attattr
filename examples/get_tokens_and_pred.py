"""Get the example input and the model prediction."""

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
from path import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling_orig import BertForSequenceClassification, BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from examples.classifier_processer import InputExample, InputFeatures, DataProcessor, MrpcProcessor, MnliProcessor, RteProcessor, ScitailProcessor, ColaProcessor, SstProcessor, QqpProcessor, QnliProcessor, WnliProcessor, StsProcessor

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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = None

    features = []
    tokens_all = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
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
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

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
                          label_id=label_id))
        tokens_all.append(tokens)
    return features, tokens_all


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
    parser.add_argument("--model_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The model file which will be evaluated.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    

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
    parser.add_argument("--num_examples",
                        default=100,
                        type=int,
                        help="The number of dev examples to compute the model prediction.")

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

    res_tokens_and_pred = []
    # Load a fine-tuned model 
    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare the data
    eval_segment = "dev_matched" if args.task_name == "mnli" else "dev"
    eval_examples = processor.get_dev_examples(
        args.data_dir, segment=eval_segment)[0:args.num_examples]

    model.eval()

    index_record_set = {}
    for label_index in range(0, len(label_list)):
        index_record_set[label_list[label_index]] = []

    eval_features, tokens_all = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation: %s *****", eval_segment)
    logger.info("  Num examples = %d", len(eval_examples))
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
        eval_data, sampler=eval_sampler, batch_size=1)

    model.eval()
    eval_loss, eval_result = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits, all_label_ids = [], []
    index_cnt = 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        tokens = tokens_all[index_cnt]

        with torch.no_grad():
            tmp_eval_loss = model(
                input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        all_logits.append(logits)
        all_label_ids.append(label_ids)
        res_tokens_and_pred.append({"index":index_cnt, "tokens":tokens, "pred":label_list[int(np.argmax(logits))], "golden":label_list[int(label_ids[0])], "pred_logits":logits.data.tolist()})

        eval_loss += tmp_eval_loss.mean().item()
        for i in range(len(label_ids)):
            for label_index in range(0, len(label_list)):
                if label_ids[i] == label_index and label_ids[i] == int(np.argmax(logits[i])):
                    index_record_set[label_list[label_index]].append(index_cnt)
                    break
            index_cnt += 1

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    with open(os.path.join(args.output_dir, "tokens_and_pred_{0}.json".format(index_cnt)), "w") as fout:
        fout.write(json.dumps(res_tokens_and_pred, indent=2) + "\n")


if __name__ == "__main__":
    main()
