# X-MAML Reproduction
I found it is a little difficult for me to re-produce the results in the EMNLP2020 paper [Zero-Shot Cross-Lingual Transfer with Meta Learning](https://aclanthology.org/2020.emnlp-main.368/) by the code in [the original repository](https://github.com/copenlu/X-MAML). So I re-write the code on my own.

## Requirements
- Python version >= 3.6
- PyTorch version == 1.6.0
- transformers version == 3.0.2
  
## Getting Started

#### 1.Finetune Multilingual Pre-trained Model
At the very begining, it is necessary to **finetune** the multilingual pretrained models (mBERT, XLM-R, etc.) on the **MNLI** dataset, which is similar to XNLI but is in English. Otherwise, the pretrained model is unable to provide reasonable embeddings in the following procedure.

My script for the finetune procedure is:
```
cd finetune

data_path=path/to/dataset/
pretrain_model_dir=../cache_dir/bert-base-multilingual-cased/

python finetune.py \
    --data_path $data_path \
    --pretrain_model_dir $pretrain_model_dir \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 128 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --seed 42
```

The finetuned model will be saved as `./finetune_saved_model/pytorch_model.bin`. 

It is worth noting that you need to copy the `*.json` and `*.txt` files in the `./cache_dir/bert-base-multilingual-cased/`, if you want to load the finetuned model through `model = BertForSequenceClassification.from_pretrained('./finetuned_saved_model')`.

You can evaluate the performance of finetuned model on the MNLI with following scripts:
```
cd finetune

data_path=path/to/dataset/
pretrain_model_dir=./finetune_saved_model

python eval_mnli.py \
    --data_path $data_path \
    --pretrain_model_dir $pretrain_model_dir \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 32 \
    --seed 42
```

If you wanna test it on the XNLI dataset, you can simply change the `eval_mnli.py` to `eval_xnli.py`.

### 2. Train the X-MAML Model
You can obtain the results under zero-shot learning with following script:
```
cd X-MAML

data_path=../../../../../data/XNLI/XNLI-1.0/
pretrain_model_dir=../finetunefinetune_saved_model/

python xmaml.py \
    --data_path $data_path \
    --pretrain_model_dir $pretrain_model_dir \
    --num_train_iter 400 \
    --num_inner_iter 1 \
    --support_size 8 \
    --query_size 8 \
    --num_tasks_in_batch 2 \
    --num_accumulation_step 1 \
    --learning_rate 2e-5 \
    --aux_langs hi \
    --seed 42 \
    --lang ru
```

Adding `--few_shot` should give you the results under few-shot learning setting.

## Results
Under the zero-shot learning setting, with `hi` as the xu language:
| seed  |  ar   |  bg   |  de   |  el   |  en   |  es   |  fr   |  hi   |  ru   |  sw   |  th   |  tr   |  ur   |  vi   |  zh   |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  42   | 65.77 | 71.35 | 73.08 | 68.36 | 82.18 | 75.41 | 75.63 |   -   | 71.51 | 48.78 | 55.75 | 63.03 | 61.53 | 72.19 | 73.26 |

There is merely some inconsistence between my results and those reported in the paper. Some further adjusting may be necessary.