# Self-Attention Attribution

This repository contains the implementation for AAAI-2021 paper [Self-Attention Attribution: Interpreting Information Interactions Inside Transformer](https://arxiv.org/pdf/2004.11207.pdf). It includes the code for generating the self-attention attribution score, pruning attention heads with our method, constructing the attribution tree and extracting the adversarial triggers. All of our experiments are conducted on bert-base-cased model, our methods can also be easily transfered to other Transformer-based models.

## Requirements
* Python version >= 3.5
* Pytorch version == 1.1.0
* networkx == 2.3

We recommend you to run the code using the docker under Linux:
```bash
docker run -it --rm --runtime=nvidia --ipc=host --privileged pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel bash
```

Then install the following packages with pip:
```bash
pip install --user networkx==2.3
pip install --user matplotlib==3.1.0
pip install --user tensorboardX six numpy tqdm scikit-learn
```

You can install attattr from source:
```bash
git clone https://github.com/YRdddream/attattr
cd attattr
pip install --user --editable .
```

## Download Pre-Finetuned Models and Datasets

Before running self-attention attribution, you can first fine-tune bert-base-cased model on a downstream task (such as MNLI) by running the file `run_classifier_orig.py`.
We also provide the example datasets and the pre-finetuned checkpoints at [Google Drive](https://drive.google.com/file/d/1L8iI2wWSF6aVtYsJ9BtdQmEGbpnpE3dW/view?usp=sharing).

## Get Self-Attention Attribution Scores
Run the following command to get the self-attention attribution score and the self-attention score.
```bash
python examples/generate_attrscore.py --task_name ${task_name} --data_dir ${data_dir} \
       --bert_model bert-base-cased --batch_size 16 --num_batch 4 \
       --model_file ${model_file} --example_index ${example_index} \
       --get_att_attr --get_att_score --output_dir ${output_dir}
```

## Construction of Attribution Tree
When you get the self-attribution scores of a target example, you could construct the attribution tree.
We recommend you to run the file `get_tokens_and_pred.py` to summarize the data, or you can manually change the value of `tokens` in `attribution_tree.py`.
```bash
python examples/attribution_tree.py --attr_file ${attr_file} --tokens_file ${tokens_file} \
       --task_name ${task_name} --example_index ${example_index} 
```
You can generate the attribution tree from the provided example.
```bash
python examples/attribution_tree.py --attr_file ${model_and_data}/mnli_example/attr_zero_base_exp16.json \
       --tokens_file ${model_and_data}/mnli_example/tokens_and_pred_100.json \
       --task_name mnli --example_index 16
```

## Self-Attention Head Pruning
We provide the code of pruning attention heads with both our attribution method and the Taylor expansion method.
Pruning heads with our method.
```bash
python examples/prune_head_with_attr.py --task_name ${task_name} --data_dir ${data_dir} \
       --bert_model bert-base-cased --model_file ${model_file}  --output_dir ${output_dir}
```
Pruning heads with Taylor expansion method.
```bash
python examples/prune_head_with_taylor.py --task_name ${task_name} --data_dir ${data_dir} \
       --bert_model bert-base-cased --model_file ${model_file}  --output_dir ${output_dir}
```


##  Adversarial Attack
First extract the most important connections from the training dataset.
```bash
python examples/run_adver_connection.py --task_name ${task_name} --data_dir ${data_dir} \
       --bert_model bert-base-cased --batch_size 16 --num_batch 4 --zero_baseline \
       --model_file ${model_file} --output_dir ${output_dir}
```

Then use these adversarial triggers to attack the original model.
```bash
python examples/run_adver_evaluate.py --task_name ${task_name} --data_dir ${data_dir} \
       --bert_model bert-base-cased --model_file ${model_file} \
       --output_dir ${output_dir} --pattern_file ${pattern_file}
```

## Reference
If you find this repository useful for your work, you can cite the paper:

```
@inproceedings{attattr,
  author = {Yaru Hao and Li Dong and Furu Wei and Ke Xu},
  title = {Self-Attention Attribution: Interpreting Information Interactions Inside Transformer},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence},
  publisher = {{AAAI} Press},
  year      = {2021},
  url       = {https://arxiv.org/pdf/2004.11207.pdf}
}
```
