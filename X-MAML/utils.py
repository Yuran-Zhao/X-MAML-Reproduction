import os
import pdb
import json
import random
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

# import sys

# sys.path.append('..')
# from finetune import XNLIDataset

LABEL2IDX = {"neutral": 0, "entailment": 1, "contradiction": 2}


class MAMLDataset:
    """Need to generate several tasks,
    each task can be further devided as:
        support set: N class and k shot, used for update the parameters in inner loop
        query set: N class and q shot, used for compute the loss with the final parameters after inner loop
    after these, the model is updated according to the whole loss computed on query set
    
    """

    def __init__(
        self,
        file_path,
        aux_langs=["hi"],
        num_tasks_in_batch=4,
        support_size=8,
        query_size=1,
    ):
        # super(MAMLDataset, self).__init__()

        cached_path = file_path + ".cached"
        if os.path.exists(cached_path):
            self._load_dataset_from_cachefile(
                cached_path, aux_langs, support_size + query_size
            )
        else:
            cached_datasets = {}
            with open(file_path, "r") as fin:
                for line in tqdm(fin):
                    data = json.loads(line)
                    # if lang != 'all' and data['language'] != lang:
                    #     continue
                    cur_lang = data["language"]
                    if cur_lang not in cached_datasets:
                        cached_datasets[cur_lang] = {
                            "sents1": [],
                            "sents2": [],
                            "labels": [],
                        }
                    sent1 = data["sentence1"]
                    sent2 = data["sentence2"]
                    label = LABEL2IDX[data["gold_label"]]
                    cached_datasets[cur_lang]["sents1"].append(sent1)
                    cached_datasets[cur_lang]["sents2"].append(sent2)
                    cached_datasets[cur_lang]["labels"].append(label)
            with open(cached_path, "wb") as fout:
                pickle.dump(cached_datasets, fout)
            self._load_dataset_from_cachefile(
                cached_path, aux_langs, support_size + query_size
            )

        self._shuffle()

        self.support_size = support_size
        self.query_size = query_size
        self.num_tasks_in_batch = num_tasks_in_batch

        self._generate_tasks(support_size + query_size)

        self.idx = 0

        self.cached_batch = self._sample_batch()

    def _load_dataset_from_cachefile(self, cached_file, aux_langs, per_task_size):
        with open(cached_file, "rb") as fin:
            cached_datasets = pickle.load(fin)
        # have to gurantee that the size of dataset can be devided by `per_task_size`
        self.sents1, self.sents2, self.labels = [], [], []
        self.task_num = 0
        for lang in aux_langs:
            cur_length = len(cached_datasets[lang]["sents1"])

            self.task_num += cur_length // per_task_size
            keep = (cur_length // per_task_size) * per_task_size

            self.sents1.extend(cached_datasets[lang]["sents1"][:keep])
            self.sents2.extend(cached_datasets[lang]["sents2"][:keep])
            self.labels.extend(cached_datasets[lang]["labels"][:keep])

    def _shuffle(self):
        # aims to gurantee that each task contains examples from different languages (if len(aux_langs) > 1)
        random.seed(1234)
        tmp_all = list(zip(self.sents1, self.sents2, self.labels))
        random.shuffle(tmp_all)

        self.sents1 = [item[0] for item in tmp_all]
        self.sents2 = [item[1] for item in tmp_all]
        self.labels = [item[2] for item in tmp_all]

    def _generate_tasks(self, per_task_size):
        self.tasks = []
        for idx in range(self.task_num):
            sents1 = self.sents1[idx * per_task_size : (idx + 1) * per_task_size]
            sents2 = self.sents2[idx * per_task_size : (idx + 1) * per_task_size]
            labels = self.labels[idx * per_task_size : (idx + 1) * per_task_size]
            self.tasks.append({"sents1": sents1, "sents2": sents2, "labels": labels})

    def _sample_batch(self):
        batch = []
        for _ in range(10):
            task = []
            sampled_ids = np.random.choice(
                self.task_num, self.num_tasks_in_batch, False
            )
            for id in sampled_ids:
                task.append(self.tasks[id])
            batch.append(task)
        return batch

    def next(self):
        if self.idx >= len(self.cached_batch):
            self.idx = 0
            self.cached_batch = self._sample_batch()
        next_batch = self.cached_batch[self.idx]
        self.idx += 1
        return next_batch


def preprocess_dataset(in_path, out_path):
    """Read data from `in_path`, which is a `.jsonl`
        reassemble them as a two-level dictionary according to `language`

    Args:
        in_path (os.path): Path to the originial `.jsonl` file
        out_path (os.path): Path to the cached file
    """
    cached_datasets = {}
    with open(in_path, "r") as fin:
        for line in tqdm(fin):
            data = json.loads(line)
            # if lang != 'all' and data['language'] != lang:
            #     continue
            cur_lang = data["language"]
            if cur_lang not in cached_datasets:
                cached_datasets[cur_lang] = {"sents1": [], "sents2": [], "labels": []}
            sent1 = data["sentence1"]
            sent2 = data["sentence2"]
            label = LABEL2IDX[data["gold_label"]]
            cached_datasets[cur_lang]["sents1"].append(sent1)
            cached_datasets[cur_lang]["sents2"].append(sent2)
            cached_datasets[cur_lang]["labels"].append(label)
    with open(out_path, "wb") as fout:
        pickle.dump(cached_datasets, fout)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_class", default=3, type=int)
    parser.add_argument("--pretrain_model_dir", default="bert-base-uncased", type=str)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--aux_langs", nargs="+", type=str)
    parser.add_argument("--src_lang", default="en", type=str)
    parser.add_argument("--dev_lang", default="es", type=str)
    parser.add_argument("--tgt_lang", default="fr", type=str)

    parser.add_argument(
        "--low_resource",
        action="store_true",
        help="Train the model with 64 examples for each lang in `aux_langs`.",
    )
    parser.add_argument(
        "--include",
        action="store_true",
        help="Add the `src_lang` `train.txt` into training dataset.",
    )

    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--num_train_iter", default=50, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1, type=float)

    parser.add_argument("--early_stopping_patience", default=3, type=int)
    parser.add_argument("--svae_iters", default=100, type=int)
    parser.add_argument("--logging_iters", default=100, type=int)
    parser.add_argument("--output_model_dir", default="./maml_saved_model", type=str)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Finetune the model on the `dev` set of target language before evaluating performance on `test` set",
    )
    parser.add_argument("--few_shot_learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_few_shot_epochs", default=1, type=int)

    parser.add_argument("--num_tasks_in_batch", default=4, type=int)
    parser.add_argument("--num_accumulation_step", default=4, type=int)
    parser.add_argument("--support_size", default=8, type=int)
    parser.add_argument("--query_size", default=8, type=int)
    parser.add_argument("--num_inner_iter", default=5, type=int)
    parser.add_argument("--inner_learning_rate", default=1e-4, type=float)
    parser.add_argument("--min_learning_rate", default=1e-6, type=float)

    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--lang", default="ar", type=str)
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--log_loss_steps", default=100, type=int)
    parser.add_argument(
        "--multiplier", default=1, type=int, help="Used to amplify the loss."
    )

    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        help="'train': train the model from scratch; 'eval': evaluate on the `tgt_lang`.",
    )
    args = parser.parse_args()

    return args
