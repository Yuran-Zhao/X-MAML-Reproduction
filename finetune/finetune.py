import argparse
import random
import time
import pdb
import os
import logging

from transformers import (
    BertTokenizer,
    BertConfig,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils import arg_parser, MultiNLIDataset

writer = SummaryWriter("runs/finetune_mBERT_on_MultiNLI_1")

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
        _, cls_out = self.bert(
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


def save_model(model, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    logger.info("Saving model to {output_path}")
    torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))


def main():
    args = arg_parser()

    seed_everything(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_dir)

    train_dataset = MultiNLIDataset(
        os.path.join(args.data_path, "multinli_1.0_train.jsonl")
    )
    eval_dataset_matched = MultiNLIDataset(
        os.path.join(args.data_path, "multinli_1.0_dev_matched.jsonl")
    )
    eval_dataset_mismatched = MultiNLIDataset(
        os.path.join(args.data_path, "multinli_1.0_dev_mismatched.jsonl")
    )
    eval_dataset = eval_dataset_matched + eval_dataset_mismatched

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

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_gpu_train_batch_size,
        collate_fn=tokenize_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.per_gpu_eval_batch_size, collate_fn=tokenize_fn
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CustomModel(args.pretrain_model_dir).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    total_steps = args.num_train_epochs * len(train_dataloader)

    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    save_steps = total_steps // args.num_train_epochs
    eval_steps = save_steps
    log_loss_steps = args.logging_steps
    avg_loss = 0.0
    best_f1 = 0.0

    for epoch in range(args.num_train_epochs):
        train_loss = 0.0
        logger.info("\n------------epoch:{}------------".format(epoch))
        last = time.time()
        for step, batch_data in enumerate(train_dataloader):
            model.train()
            batch_data = {k: v.to(device) for k, v in batch_data.items()}

            loss = model(**batch_data, loss_fn=loss_fn)[1]

            loss.backward()

            train_loss += loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            model.zero_grad()

            global_step += 1
            if global_step % log_loss_steps == 0:
                avg_loss /= log_loss_steps
                logger.info(
                    "Step: %d / %d ----> total loss: %.5f"
                    % (global_step, total_steps, avg_loss)
                )
                writer.add_scalar("train/loss", avg_loss, global_step)
                avg_loss = 0.0
            else:
                avg_loss += loss.item()
        logger.info(f"[{epoch} / {args.num_train_epochs}]: {time.time() - last}")

        eval_loss = 0
        eval_acc = 0
        y_true = []
        y_predict = []
        y_predict_target = []

        model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(eval_dataloader):
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(device)
                labels = batch_data["labels"]
                y_true.extend(labels.cpu().numpy())

                logits, loss = model(**batch_data, loss_fn=loss_fn)
                predict_scores = F.softmax(logits)
                y_predict_target.extend(
                    predict_scores.argmax(dim=1).detach().to("cpu").numpy()
                )
                predict_scores = predict_scores[:, 1]
                y_predict.extend(predict_scores.detach().to("cpu").numpy())

                acc = ((logits.argmax(dim=-1) == labels).sum()).item()
                eval_acc += acc / logits.shape[0]
                eval_loss += loss

        eval_loss = eval_loss / len(eval_dataloader)
        eval_acc = eval_acc / len(eval_dataloader)
        eval_f1 = f1_score(y_true, y_predict_target, average="macro")
        writer.add_scalar("eval/loss", eval_loss, epoch)
        writer.add_scalar("eval/acc", eval_acc, epoch)
        writer.add_scalar("eval/f1", eval_f1, epoch)

        if best_f1 < eval_f1:
            early_stop = 0
            best_f1 = eval_f1
            save_model(model, args.output_model_dir)
        else:
            early_stop += 1

        logger.info(
            "epoch: %d, train loss: %.8f, eval loss: %.8f, eval acc: %.8f, eval f1: %.8f, best_f1: %.8f\n"
            % (epoch, train_loss, eval_loss, eval_acc, eval_f1, best_f1)
        )

        torch.cuda.empty_cache()

        if early_stop >= args.early_stopping_patience:
            break


if __name__ == "__main__":
    main()
