from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import random
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score
from scipy.stats import pearsonr


def pred_argmax(out):
    return np.argmax(out, axis=1).reshape(-1)


def accuracy(out, labels):
    outputs = pred_argmax(out)
    r = accuracy_score(labels.reshape(-1), outputs)
    if np.isnan(r):
        r = 0.0
    return float(r)


def mcc(out, labels):
    outputs = pred_argmax(out)
    r = matthews_corrcoef(labels.reshape(-1), outputs)
    if np.isnan(r):
        r = 0.0
    return float(r)


def pearson_cc(out, labels):
    r = pearsonr(labels.reshape(-1), out.reshape(-1))[0]
    if np.isnan(r):
        r = 0.0
    return float(r)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, baseline_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.baseline_ids = baseline_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, segment='train'):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, segment='dev'):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, segment='test'):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_train_segments(self):
        return ['train']

    def get_dev_segments(self):
        return ['dev']

    def get_test_segments(self):
        return ['test']

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_metric_func(self):
        return accuracy

    def get_pred(self, out):
        # default: classification
        lbl_list = self.get_labels()
        return [lbl_list[p] for p in pred_argmax(out).tolist()]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, segment='train'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-2]
            text_b = line[-1]
            if is_test:
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, segment='train'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev_matched'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, segment+".tsv")), segment)

    def get_test_examples(self, data_dir, segment='test_matched'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, segment+".tsv")), segment, is_test=True)

    def get_dev_segments(self):
        return ['dev_matched', 'dev_mismatched']

    def get_test_segments(self):
        return ['test_matched', 'test_mismatched', 'diagnostic']

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = self.get_labels()[0]
            else:
                text_a = line[8]
                text_b = line[9]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if is_test:
                label = self.get_labels()[0]
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-1]
            if is_test:
                label = self.get_labels()[0]
            else:
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_metric_func(self):
        return mcc


class SstProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, segment='train'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if is_test:
                text_a = line[1]
                label = self.get_labels()[0]
            else:
                text_a = line[0]
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def _read_input_file(self, input_file):
        """Reads a tab separated value file."""
        lines = []
        with open(input_file, "r", encoding='utf-8') as f:
            for l in f:
                col_list = l.strip().split('\t')
                if len(col_list) == 6:
                    lines.append(col_list)
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_input_file(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_input_file(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = self.get_labels()[0]
            else:
                if len(line) != 6:
                    print('Skip:', line)
                    continue
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            text_a = text_a.strip("\"")
            text_b = text_b.strip("\"")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = self.get_labels()[0]
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = self.get_labels()[0]
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class StsProcessor(DataProcessor):
    """Processor for the STS data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return None

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = str(random.random())
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_metric_func(self):
        return pearson_cc

    def get_pred(self, out):
        return [p for p in out.reshape(-1).tolist()]


class ScitailProcessor(DataProcessor):
    """Processor for the Scitail data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "scitail_1.0_train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "scitail_1.0_dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "scitail_1.0_test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["neutral", "entails"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
