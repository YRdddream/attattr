"""Get the most important attention connections."""

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
from pytorch_pretrained_bert.model_prune_head import BertForSequenceClassification, BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from examples.classifier_processer_tmp import InputExample, InputFeatures, DataProcessor, MrpcProcessor, MnliProcessor, RteProcessor, ScitailProcessor, ColaProcessor, SstProcessor, QqpProcessor, QnliProcessor, WnliProcessor, StsProcessor

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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, data_name="mnli"):
    """Loads a data file into a list of `InputBatch`s."""
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = None

    features = []
    tokenslist = []
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
        base_tokens = ["[UNK]"] + ["[UNK]"]*len(tokens_a) + ["[UNK]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            base_tokens += ["[UNK]"]*len(tokens_b) + ["[UNK]"]
            segment_ids += [1] * (len(tokens_b) + 1)

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

        if data_name == "sst-2" and len(tokens) < 15:
            continue
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          tokens=tokens,
                          baseline_ids=baseline_ids))
        tokenslist.append({"token":tokens, "golden_label":example.label, "pred_label":None})
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

def scaled_input(emb, batch_size, num_batch, baseline=None):
    # emb shape: [num_head, seq_len, seq_len]
    if baseline is None:
        baseline = torch.zeros_like(emb)   

    num_points = batch_size * num_batch
    scale = 1.0 / num_points

    step = (emb.unsqueeze(0) - baseline.unsqueeze(0)) * scale
    res = torch.cat([torch.add(baseline.unsqueeze(0), step*i) for i in range(num_points)], dim=0)
    return res, step[0]


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

    # get adversatial connections parameters
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=4,
                        type=int,
                        help="Num batch of an example.")
    parser.add_argument("--zero_baseline",
                        default=False,
                        action='store_true',
                        help="If use zero atteniton matrix as the baseline.")
    parser.add_argument("--start_exp",
                        default=0,
                        type=int,
                        help="The start index of training examples.")
    parser.add_argument("--num_exp",
                        default=500,
                        type=int,
                        help="The number of training examples for finding patterns.")
    parser.add_argument("--data_type",
                        default="train",
                        type=str,
                        help="Patterns from dev_set or training_set.")


    args = parser.parse_args()
    args.zero_baseline = True

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

    # Load a fine-tuned model 
    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare the data
    # dev_examples or train_examples
    if args.data_type == "dev":
        eval_segment = "dev_matched" if args.task_name == "mnli" else "dev"
        eval_examples = processor.get_dev_examples(
            args.data_dir, segment=eval_segment)[args.start_exp:args.start_exp+args.num_exp]
    else:   # train set
        eval_segment = "train"
        eval_examples = processor.get_train_examples(args.data_dir)[args.start_exp:args.start_exp+args.num_exp]

    model.eval()

    if args.bert_model.find("base") != -1:
        num_head, num_layer = 12, 12
    elif args.bert_model.find("large") != -1:
        num_head, num_layer = 16, 24

    eval_loss, eval_result = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits, all_label_ids = [], []
    seg_result_dict = {}

    saved_res = []

    eval_features, _ = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.task_name)
    logger.info("***** Running evaluation: %s *****", eval_segment)
    logger.info("  Num examples = %d", len(eval_examples))
    all_baseline_ids = torch.tensor(
        [f.baseline_ids for f in eval_features], dtype=torch.long)
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=lbl_type)
    all_tokens = [f.tokens for f in eval_features]

    eval_data = TensorDataset(
        all_baseline_ids, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction 
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=1)
    
    model.eval()
    eval_loss, eval_result = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits, all_label_ids = [], []
    index_count = 0
    for baseline_ids, input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        input_len = int(input_mask[0].sum())       

        tokens = all_tokens[index_count]
        seg_pos = tokens.index("[SEP]")
        tar_head_attr = None

        with torch.no_grad():
            tmp_eval_loss, baseline_logits = model(input_ids, "res", segment_ids, input_mask, label_ids)
        pred_label = int(torch.argmax(baseline_logits))
        attr_max = None   

        for tar_layer in range(0, num_layer):
            with torch.no_grad():
                att, _ = model(input_ids, "att", segment_ids, input_mask, label_ids, tar_layer)
            att = att[0]
            baseline = None
            scale_att, step = scaled_input(att.data, args.batch_size, args.num_batch, baseline)
            scale_att.requires_grad_(True)

            attr_all = None
            for j_batch in range(args.num_batch):
                one_batch_att = scale_att[j_batch*args.batch_size:(j_batch+1)*args.batch_size]
                tar_prob, grad = model(input_ids, "att", segment_ids, input_mask, label_ids, 
                                        tar_layer, one_batch_att, pred_label=pred_label)
                grad = grad.sum(dim=0)  
                attr_all = grad if attr_all is None else torch.add(attr_all, grad)
            attr_all = attr_all[:,0:input_len,0:input_len] * step[:,0:input_len,0:input_len]  
            tar_head_index = int(torch.argmax(attr_all.reshape(num_head*input_len*input_len))) // (input_len*input_len)
            if tar_head_attr is None:
                tar_head_attr = attr_all[tar_head_index]
            else:
                if attr_all[tar_head_index].max() > tar_head_attr.max():
                    tar_head_attr = attr_all[tar_head_index]

        attr_max = tar_head_attr.cpu().numpy().reshape(input_len*input_len)
        attr_sorted_index = np.argsort(attr_max)[::-1]   # reverse order
        saved_res.append({"max_combined_attr":[float(attr_max[attr_sorted_index[0]]), float(attr_max[attr_sorted_index[1]])], 
                            "top1pattern":[tokens[attr_sorted_index[0]//input_len], tokens[attr_sorted_index[0]%input_len]],
                            "top2pattern":[tokens[attr_sorted_index[1]//input_len], tokens[attr_sorted_index[1]%input_len]],
                            "top1position":[int(attr_sorted_index[0]//input_len), int(attr_sorted_index[0]%input_len)],
                            "top2position":[int(attr_sorted_index[1]//input_len), int(attr_sorted_index[1]%input_len)],
                            "target_label":pred_label,
                            "golden_label":int(label_ids[0]),
                            "seg_pos":seg_pos,
                            "tokens":tokens})

        logits = baseline_logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        all_logits.append(logits)
        all_label_ids.append(label_ids)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
        index_count += 1
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
    if eval_segment not in seg_result_dict:
        seg_result_dict[eval_segment] = []
    seg_result_dict[eval_segment].append(result)
    logger.info("***** Eval results ({0}) *****".format(eval_segment))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    
    saved_res = sorted(saved_res, key=lambda a: sum(a["max_combined_attr"]), reverse=True)
    with open(os.path.join(args.output_dir, "{0}_adver_pattern_exp{1}-{2}.json".format(args.data_type, args.start_exp, args.start_exp+args.num_exp)), "w") as fout:
        fout.write(json.dumps(saved_res, indent=2) + "\n")


if __name__ == "__main__":
    main()