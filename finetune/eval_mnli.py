import argparse
import random
import time
import pdb
import os
import logging

# import sys
# sys.path.append("../../..")
# from xnli_transformers import BertConfig
# from transformers import BertTokenizer
# from BERT.bert import BertForSequenceClassification
from transformers import (
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .utils import arg_parser, MultiNLIDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomModel(nn.Module):
    def __init__(self, pretrain_model_path):
        super(CustomModel, self).__init__()
        self.config = BertConfig.from_pretrained(
            pretrain_model_path, output_hidden_states=True
        )
        print(pretrain_model_path)
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, 3)

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels=None, loss_fn=None
    ):
        sequence_out, cls_out = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        cls_out = self.dropout(cls_out)
        logits = self.classifier(cls_out)
        if loss_fn is not None:
            loss = loss_fn(logits, labels)
            return logits, loss
        return logits


def main():
    args = arg_parser()

    seed_everything(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_dir)

    eval_dataset = MultiNLIDataset(
        os.path.join(args.data_path, "multinli_1.0_dev_matched.jsonl")
    )

    def tokenize_fn(examples):
        alls = [
            tokenizer(
                example["sent1"],
                example["sent2"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )
            for example in examples
        ]
        input_ids = [cur["input_ids"] for cur in alls]
        token_type_ids = [cur["token_type_ids"] for cur in alls]
        attention_mask = [cur["attention_mask"] for cur in alls]
        labels = [example["label"] for example in examples]
        return {
            "input_ids": torch.tensor(input_ids),
            "token_type_ids": torch.tensor(token_type_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.per_gpu_eval_batch_size, collate_fn=tokenize_fn
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained(
        args.pretrain_model_dir, num_labels=3, cache_dir=args.pretrain_model_dir
    )
    model = BertForSequenceClassification(config)
    model.from_pretrained(args.pretrain_model_dir)
    model = model.to(device)

    eval_acc = 0.0
    y_true = []
    y_predict_target = []
    y_predict = []
    model.eval()
    for step, batch_data in enumerate(eval_dataloader):
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(device)
        labels = batch_data["labels"]
        y_true.extend(labels.cpu().numpy())

        _, logits = model(**batch_data)
        predict_scores = F.softmax(logits)
        y_predict_target.extend(predict_scores.argmax(dim=1).detach().to("cpu").numpy())
        predict_scores = predict_scores[:, 1]
        y_predict.extend(predict_scores.detach().to("cpu").numpy())

        acc = ((logits.argmax(dim=-1) == labels).sum()).item()
        eval_acc += acc / logits.shape[0]

    eval_acc = eval_acc / len(eval_dataloader)
    eval_f1 = f1_score(y_true, y_predict_target, average="macro")

    logger.info(f"acc: {eval_acc} | f1: {eval_f1}")


if __name__ == "__main__":
    main()
