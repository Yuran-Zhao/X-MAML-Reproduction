import argparse
import random
import time
import pdb
import os
import logging

from transformers import (
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import arg_parser, XNLIDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = arg_parser()

    seed_everything(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_dir)

    dev_dataset = XNLIDataset(os.path.join(args.data_path, "xnli.dev.jsonl"), args.lang)
    test_dataset = XNLIDataset(
        os.path.join(args.data_path, "xnli.test.jsonl"), args.lang
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

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.per_gpu_eval_batch_size, collate_fn=tokenize_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.per_gpu_eval_batch_size, collate_fn=tokenize_fn
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        args.pretrain_model_dir, num_labels=3
    )
    model = model.to(device)

    if args.few_shot:
        logger.info("\nFurther finetune on `dev` set")

        model = further_finetune(model, dev_dataloader, test_dataloader, device, args)
    logger.info("\nFinal Evaluation on `test` set")
    eval(model, test_dataloader, device, args.lang)


def further_finetune(model, train_dataloader, dev_dataloader, device, args):
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
        optimizer_grouped_parameters,
        lr=args.few_shot_learning_rate,
        eps=args.adam_epsilon,
    )

    total_steps = args.num_few_shot_epochs * len(train_dataloader)

    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    # loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    save_steps = total_steps // args.num_few_shot_epochs
    eval_steps = save_steps
    log_loss_steps = args.logging_steps
    avg_loss = 0.0
    best_f1 = 0.0

    for epoch in range(args.num_few_shot_epochs):
        train_loss = 0.0
        logger.info("\n------------epoch:{}------------".format(epoch))
        last = time.time()
        for step, batch_data in enumerate(train_dataloader):

            model.train()
            batch_data = {k: v.to(device) for k, v in batch_data.items()}

            # loss = model(**batch_data, loss_fn=loss_fn)[1]
            loss, logits = model(**batch_data)

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
                avg_loss = 0.0
            else:
                avg_loss += loss.item()
        logger.info(f"[{epoch} / {args.num_few_shot_epochs}]: {time.time() - last}")

        eval(model, dev_dataloader, device, args.lang)
    return model


def eval(model, dataloader, device, lang):
    eval_acc = 0.0
    y_true = []
    y_predict_target = []
    y_predict = []
    model.eval()
    for step, batch_data in enumerate(dataloader):
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

    eval_acc = eval_acc / len(dataloader)
    eval_f1 = f1_score(y_true, y_predict_target, average="macro")

    logger.info(f"[ {lang.upper()} ]acc: {eval_acc} | f1: {eval_f1}")


if __name__ == "__main__":
    main()
