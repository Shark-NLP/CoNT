from train import str2bool
import os
import argparse
import random

WARM_UP_PATH = "pretrained_weights/java/"


def run(inp_cmd):
    print(inp_cmd)
    os.system(inp_cmd)


# Please set `--dataset python` if you want to reproduce our results on python 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "test", "val"])
    parser.add_argument('--warmup', type=str2bool, default=True,
                        help="if you set warmup=False ensure `WARM_UP_PATH` not empty")
    parser.add_argument('--gpus', default="0,1,2,3,4,5,6,7")
    parser.add_argument('--model_name', default="Salesforce/codet5-base", choices=["Salesforce/codet5-base"])
    parser.add_argument('--accum_count', default=1)
    parser.add_argument('--warmup_batch_size', default=32)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--dataset', default="java", choices=["java", "python"])
    parser.add_argument('--validate_every', default=800, type=int)
    # no need to set in training mode
    parser.add_argument('--save_path', default="")  # dir/contains/checkpoints
    args = parser.parse_args()

    ptm = args.model_name.split("/")[-1].split("-")[0]
    print("You are using the pretrain model: ", ptm)

    args.model_name = args.model_name.replace("java", args.dataset)
    WARM_UP_PATH = WARM_UP_PATH.replace("java", args.dataset)

    # if your model name contains the dataset name  `java/python`, we will skip the warmup
    base_model_cont = WARM_UP_PATH + ptm
    if args.dataset in args.model_name.lower():
        args.warmup = False
        base_model_cont = args.model_name

    inference_param = " --alpha 0.2  --min_length 0 --max_length 128 --length_pen 0.6 "

    if args.mode != "train":
        test_cmd = f"python inference.py --gpus {args.gpu}  --dataset {args.dataset} " \
                   f" --warmup True --mode {args.mode} --batch_size {args.batch_size} " \
                   f" --model_name {args.model_name}  --save_path {args.save_path}  --PTM {ptm} " \
                   f" --diversity_pen 0.0 --beam_size 8  {inference_param} "
        run(test_cmd)
    else:
        num_process = len(args.gpus.split(','))
        train_cmd = f"python train.py --gpus {args.gpus} --max_src_len 512 --max_tgt_len 128 --mode train --lr 1e-3  " \
                    f" --dataset {args.dataset} --PTM {ptm} --model_name {args.model_name} --accum_count {args.accum_count} " \
                    f" --diversity_pen 2.0 --beam_size 12 {inference_param} "
        if args.warmup:
            train_cmd += f" --warmup True --batch_size {args.warmup_batch_size} " \
                         f" --n_epochs 20 --validate_every {args.validate_every} --model_name {args.model_name} " \
                         f" --save_path {WARM_UP_PATH + ptm} "
            run(train_cmd)

        train_cmd += f" --warmup False --batch_size {args.batch_size} --lr 2e-5 -n_epochs 5 " \
                     f" --validate_every {args.validate_every // 4} --reset_optimizer True --model_name {base_model_cont}" \
                     f" --save_path checkpoints/{args.dataset}/{ptm} "
        run(train_cmd)
