import os
import pdb
import time
import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import autocast as autocast

import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, AdamW

import higher

from utils import arg_parser, MAMLDataset

import sys

sys.path.append('..')
from finetune import eval, XNLIDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def maml_train(model, tokenizer, train_dataset, optimizer, device, args):
    model.train()
    support_size = args.support_size
    total_step = 0
    query_losses = []
    query_accs = []

    def collate_fn(tasks):
        rets = []
        for task in tasks:
            # task is a dict, containing several examples
            examples = list(zip(task['sents1'], task['sents2'], task['labels']))
            alls = [
                tokenizer(example[0],
                          example[1],
                          truncation=True,
                          max_length=args.max_seq_length,
                          padding='max_length') for example in examples
            ]
            input_ids = [cur['input_ids'] for cur in alls]
            token_type_ids = [cur['token_type_ids'] for cur in alls]
            attention_mask = [cur['attention_mask'] for cur in alls]
            labels = [example[2] for example in examples]
            support_set = {
                'input_ids': torch.tensor(input_ids[:support_size]),
                'token_type_ids': torch.tensor(token_type_ids[:support_size]),
                'attention_mask': torch.tensor(attention_mask[:support_size]),
                'labels': torch.tensor(labels[:support_size])
            }
            query_set = {
                'input_ids': torch.tensor(input_ids[support_size:]),
                'token_type_ids': torch.tensor(token_type_ids[support_size:]),
                'attention_mask': torch.tensor(attention_mask[support_size:]),
                'labels': torch.tensor(labels[support_size:])
            }
            rets.append((support_set, query_set))
        return rets

    for it in range(args.num_train_iter):
        start_time = time.time()
        num_inner_iter = args.num_inner_iter

        tasks = train_dataset.next()
        # pdb.set_trace()
        processed_task = collate_fn(tasks)
        task_num = len(processed_task)

        inner_optimizer = torch.optim.SGD(model.parameters(),
                                          lr=args.inner_learning_rate)
        # task_sampler = RandomSampler(train_dataset)
        # task_dataloader = DataLoader(train_dataset,
        #                              sampler=task_sampler,
        #                              batch_size=int(args.num_tasks_in_batch *
        #                                             args.num_accumulation_step),
        #                              collate_fn=collate_fn)
        optimizer.zero_grad()  # TODO: shouled we use model.zero_grad() ?

        for idx in range(task_num):
            support_batch, query_batch = processed_task[idx]

            with higher.innerloop_ctx(
                    model, inner_optimizer,
                    copy_initial_weights=False) as (fast_model, diffopt):
                for _ in range(num_inner_iter):
                    batch_data = {
                        k: v.to(device)
                        for k, v in support_batch.items()
                    }
                    with autocast():
                        support_loss, _ = fast_model(**batch_data)
                    diffopt.step(support_loss)
                    del batch_data
                batch_data = {k: v.to(device) for k, v in query_batch.items()}
                with autocast():
                    query_loss, query_logits = fast_model(**batch_data)
                query_losses.append(query_loss.detach())
                # pdb.set_trace()
                query_acc = (query_logits.argmax(dim=1).cpu()
                             == query_batch['labels']).sum().item() / len(
                                 query_batch['input_ids'])
                query_accs.append(query_acc)

                query_loss.backward()
                del batch_data
            total_step += args.query_size

        if total_step % (args.num_accumulation_step * args.query_size) == 0:
            optimizer.step()

            query_losses = sum(query_losses) / len(query_losses)
            query_accs = 100. * sum(query_accs) / len(query_accs)
            iter_time = time.time() - start_time

            print(
                f"[Iteration {it // args.num_accumulation_step} / {args.num_train_iter // args.num_accumulation_step}] Train Loss: {query_losses:.2f} | Acc: {query_accs:.2f} | Time: {iter_time:.2f}"
            )

            query_losses = []
            query_accs = []

        torch.cuda.empty_cache()

        if it // args.num_accumulation_step % args.logging_iters == 0:
            test(model, tokenizer, device, args)

    return model


def test(model, tokenizer, device, args):
    test_dataset = XNLIDataset(os.path.join(args.data_path, 'xnli.test.jsonl'),
                               args.lang)

    def tokenize_fn(examples):
        alls = [
            tokenizer(example['sent1'],
                      example['sent2'],
                      truncation=True,
                      max_length=args.max_seq_length,
                      padding='max_length') for example in examples
        ]
        input_ids = [cur['input_ids'] for cur in alls]
        token_type_ids = [cur['token_type_ids'] for cur in alls]
        attention_mask = [cur['attention_mask'] for cur in alls]
        labels = [example['label'] for example in examples]
        return {
            'input_ids': torch.tensor(input_ids),
            'token_type_ids': torch.tensor(token_type_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.per_gpu_eval_batch_size,
                                 collate_fn=tokenize_fn)

    eval(model, test_dataloader, device, args.lang)


def main():
    args = arg_parser()
    seed_everything(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_dir)
    train_dataset = MAMLDataset(os.path.join(args.data_path, 'xnli.dev.jsonl'),
                                args.aux_langs,
                                num_tasks_in_batch=args.num_tasks_in_batch,
                                support_size=args.support_size,
                                query_size=args.query_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        args.pretrain_model_dir, num_labels=3)
    model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)

    model = maml_train(model, tokenizer, train_dataset, optimizer, device, args)

    test(model, tokenizer, device, args)


if __name__ == '__main__':
    main()