import argparse
import glob
import json
from multiprocessing import Pool

import torch
from tqdm import tqdm
from transformers import PegasusTokenizer, T5Tokenizer, AutoTokenizer
import os
from train import str2bool
from model.model import CoNTGenerator
from preprocess.preprocess import JSONL_FILE_DIR
import warnings

warnings.filterwarnings("ignore")

prompt = {"wiki_bio": "convert the table to text: ",
          "totto_meta": "",
          "common_gen": "generate a sentence with: ",
          "multi_news": "summarize: ",
          "xsum": "summarize: ",
          "wmt16_ROEN": "translate Romanian to English: ",
          "java": "<java> ",
          "python": "<python> "
          }


def generate_batch(sources, targets, article_ids, model, device, sys_file, ref_file):
    dct = tokenizer.batch_encode_plus(sources, max_length=args.max_src_len, return_tensors="pt", truncation=True,
                                      padding=True)
    text_id = dct["input_ids"]
    batch_size, seq_len = text_id.size(0), text_id.size(1)
    for p in model.parameters():
        p.requires_grad = False
    cand_id = model.generate(
        input_ids=text_id.to(device),
        attention_mask=dct["attention_mask"].to(device),
        args=args
    )
    sys_outs = []
    ref_outs = []
    for i in range(batch_size):
        dec = tokenizer.decode(cand_id[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        sys_outs.append(dec)
        ref_outs.append(targets[i])

    with open(sys_file, "a") as f:
        for i in range(len(article_ids)):
            inst = {"article_id": article_ids[i], "sys_out": sys_outs[i]}
            print(json.dumps(inst), file=f)
    with open(ref_file, "a") as f:
        for i in range(len(article_ids)):
            inst = {"article_id": article_ids[i], "ref_out": ref_outs[i]}
            print(json.dumps(inst), file=f)


def load_jsonl(path):
    inst_list = []
    begin_idx = 0
    with open(path) as f:
        for line in f:
            inst = json.loads(line)
            inst["article_id"] = str(begin_idx)
            begin_idx += 1
            inst_list.append(inst)
    return inst_list


def split_dataset_into(num_parts=8):
    insts = load_jsonl(test_file)
    insts_every_ds = len(insts) // num_parts
    new_insts = []
    for i in range(num_parts + 1):
        new_insts.append(insts[i * insts_every_ds:(i + 1) * insts_every_ds])
    last_inst = new_insts.pop()
    new_insts[-1].extend(last_inst)
    assert len(new_insts) == num_parts
    return new_insts


def gen_sys_out(inp_args):
    insts_index, ckpt_path = inp_args
    insts = insts_split[insts_index]
    insts_every_ds = args.batch_size
    num_batches = len(insts) // insts_every_ds + 1
    new_insts = []
    for i in range(num_batches):
        insts_batch = insts[i * insts_every_ds:(i + 1) * insts_every_ds]
        new_insts.append(insts_batch)
    device = f'cuda:{gpus[insts_index]}'
    model = torch.load(ckpt_path, map_location=torch.device('cpu'))
    sys_file = ckpt_path.replace("checkpoints", "results").replace(".pt", f".{args.mode}.sys")
    ref_file = sys_file.replace(".sys", ".ref")
    model.eval()
    model.to(device)
    if os.path.exists(sys_file):
        os.remove(sys_file)
    if os.path.exists(ref_file):
        os.remove(ref_file)
    with tqdm(total=len(new_insts)) as pbar:
        for insts_batch in new_insts:
            if "t5" in args.PTM:
                sources = [prompt[args.dataset] + inst_batch["source"] for inst_batch in insts_batch]
            else:
                sources = [inst_batch["source"] for inst_batch in insts_batch]
            if len(sources) == 0:
                break
            targets = [inst_batch["target"] for inst_batch in insts_batch]
            article_ids = [inst_batch["article_id"] for inst_batch in insts_batch]
            generate_batch(sources, targets, article_ids, model, device, sys_file, ref_file)
            pbar.update(1)


# this is a multi-process decoding script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--ckpt', default="")
    parser.add_argument('--dataset', default=None, required=True)
    parser.add_argument('--gpus', default="0,1,2,3")
    parser.add_argument('--model_name', default="t5-small")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_src_len', default=512, type=int)
    parser.add_argument('--mode', choices=["test", "val"], default="test")
    parser.add_argument('--PTM', choices=["t5", "pegasus", "codet5"])

    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--early_stop', default=True, type=str2bool)
    parser.add_argument('--no_repeat_ngram', default=4, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--min_length', default=5, type=int)
    parser.add_argument('--max_length', default=150, type=int)
    parser.add_argument('--diversity_pen', default=0.0, type=float)
    parser.add_argument('--length_pen', default=2.0, type=float)

    args = parser.parse_args()
    gpus = args.gpus.split(",")
    test_file = f"{JSONL_FILE_DIR}/{args.dataset}/{args.mode}.jsonl"
    insts_split = split_dataset_into(len(gpus))
    print(f"you are using the {args.PTM} tokenizer to tokenize {test_file}....")

    if args.PTM == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-small", local_files_only=True)
    elif args.PTM == "codet5":
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    elif args.PTM == "pegasus":
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.PTM)
    if args.save_path and args.save_path.lower() != "none":
        ckpts = glob.glob(os.path.join(args.save_path, f'*{args.ckpt}.pt'))
    else:
        raise Exception("please check the parameter  `--save_path` ")
    print("=" * 20, "begin testing: ", ckpts, "=" * 20)
    for ckpt in ckpts:
        write_dir = args.save_path.replace("checkpoints", "results")
        os.makedirs(write_dir, exist_ok=True)
        with Pool(len(gpus)) as p:
            p.map(gen_sys_out, zip(range(len(gpus)), [ckpt] * len(gpus)))
