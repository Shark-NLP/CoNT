import json
import os
from transformers import AutoTokenizer
import argparse

JSONL_FILE_DIR = "jsonl_files"
TOKENIED_FILE_DIR = "tokenized_files"

T5_PROMPT = {"wiki_bio": "convert the table to text: ",
             "totto_meta": "",
             "common_gen": "generate a sentence with: ",
             "multi_news": "summarize: ",
             "xsum": "summarize: ",
             "wmt16_ro-en": "translate Romanian to English: ",
             "java": "<java> ",
             "python": "<python> "
             }


def tokenize_raw(ds_name, model="t5-small", ptm_alias="t5", prompt=""):
    """
    ds_name: file name of raw data
    model: the pretrained model used
    ptm_alias: alias for the model
    prompt: optional, prompt for t5-based model
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    base_dir = f"{JSONL_FILE_DIR}/{ds_name}"
    tokenized_dir = f"{TOKENIED_FILE_DIR}/{ds_name}"
    if not os.path.exists(tokenized_dir):
        os.makedirs(tokenized_dir)
    files = ["val.jsonl", "train.jsonl"]
    files_tokenized = [f"val.{ptm_alias}.jsonl", f"train.{ptm_alias}.jsonl"]
    insts_list = []
    for file in files:
        insts = []
        with open(os.path.join(base_dir, file)) as f:
            for line in f:
                insts.append(json.loads(line))
        insts_list.append(insts)
    for i, insts in enumerate(insts_list):
        for inst in insts:
            if "t5" in model:
                source = prompt + inst["source"]
            else:
                source = inst["source"]
            target = inst["target"]
            src_id = tokenizer.encode(source)
            tgt_id = tokenizer.encode(target)
            inst["src_id"] = src_id
            inst["tgt_id"] = tgt_id
        print("write into ... ", os.path.join(tokenized_dir, files_tokenized[i]))
        with open(os.path.join(tokenized_dir, files_tokenized[i]), "w") as f:
            for inst in insts:
                print(json.dumps(inst, ensure_ascii=False), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True,
                        help=" the pretrained model used, tokenizer.from_pretrained(model_name)")
    parser.add_argument('--dataset', required=True, help="selected dataset")
    parser.add_argument('--ptm', default=None, help=" mark the tokenized file")

    args = parser.parse_args()
    if not args.ptm:
        args.ptm = args.model_name.split("/")[-1].split("-")[0]
        print("You are using the pretrain model: ", args.ptm)
    # if you need a prompt
    prompt = ""
    if "t5" in args.model_name:
        prompt = T5_PROMPT[args.dataset]

    tokenize_raw(args.dataset, args.model_name, args.ptm, prompt)
