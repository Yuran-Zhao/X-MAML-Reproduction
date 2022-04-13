import os
import json
import pickle
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


LABEL2IDX = {"neutral": 0, "entailment": 1, "contradiction": 2}


class MultiNLIDataset(Dataset):
    def __init__(self, file_path):
        super(MultiNLIDataset, self).__init__()
        cached_path = file_path + ".cached"
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as fin:
                cached_datasets = pickle.load(fin)
            self.sents1 = cached_datasets["sents1"]
            self.sents2 = cached_datasets["sents2"]
            self.labels = cached_datasets["labels"]
        else:
            self.sents1, self.sents2 = [], []
            self.labels = []
            with open(file_path, "r", encoding="utf8") as fin:
                for line in tqdm(fin):
                    data = json.loads(line)
                    if data["gold_label"] not in LABEL2IDX:
                        continue
                    sent1 = data["sentence1"]
                    sent2 = data["sentence2"]
                    label = LABEL2IDX[data["gold_label"]]
                    self.sents1.append(sent1)
                    self.sents2.append(sent2)
                    self.labels.append(label)
            cached_datasets = {
                "sents1": self.sents1,
                "sents2": self.sents2,
                "labels": self.labels,
            }
            with open(cached_path, "wb") as fout:
                pickle.dump(cached_datasets, fout)

    def __len__(self):
        return len(self.sents1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {
            "sent1": self.sents1[idx],
            "sent2": self.sents2[idx],
            "label": self.labels[idx],
        }


class XNLIDataset(Dataset):
    def __init__(self, file_path, lang="all"):
        super(XNLIDataset, self).__init__()
        cached_path = file_path + ".cached"
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as fin:
                cached_datasets = pickle.load(fin)
            self.sents1 = cached_datasets[lang]["sents1"]
            self.sents2 = cached_datasets[lang]["sents2"]
            self.labels = cached_datasets[lang]["labels"]
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
            # cached_datasets = {'sents1': self.sents1, 'sents2': self.sents2, 'labels': self.labels}
            self.sents1 = cached_datasets[lang]["sents1"]
            self.sents2 = cached_datasets[lang]["sents2"]
            self.labels = cached_datasets[lang]["labels"]
            with open(cached_path, "wb") as fout:
                pickle.dump(cached_datasets, fout)

    def __len__(self):
        return len(self.sents1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {
            "sent1": self.sents1[idx],
            "sent2": self.sents2[idx],
            "label": self.labels[idx],
        }


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--pretrain_model_dir", default="bert-base-uncased", type=str)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--lang", default="ar", type=str)

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int)
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1, type=float)

    parser.add_argument("--early_stopping_patience", default=3, type=int)
    parser.add_argument("--svae_steps", default=100, type=int)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument(
        "--output_model_dir", default="./finetune_saved_model", type=str
    )
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Finetune the model on the `dev` set of target language before evaluating performance on `test` set",
    )
    parser.add_argument("--few_shot_learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_few_shot_epochs", default=1, type=int)

    args = parser.parse_args()

    return args
