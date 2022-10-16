from train import str2bool
import os
import argparse
import random

DATASET = "common_gen"
WARM_UP_PATH = "pretrained_weights/common_gen/"


def run(inp_cmd):
    print(inp_cmd)
    os.system(inp_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "test", "val"])
    parser.add_argument('--warmup', type=str2bool, default=True,
                        help="if you set warmup=False ensure `WARM_UP_PATH` not empty")
    parser.add_argument('--gpus', default="0,1,2,3,4,5,6,7")
    parser.add_argument('--model_name', default="t5-base", choices=["t5-base"])
    parser.add_argument('--accum_count', default=1)
    parser.add_argument('--warmup_batch_size', default=32)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--validate_every', default=100, type=int)

    # no need to set in training mode
    parser.add_argument('--save_path', default="")  # dir/contains/checkpoints
    args = parser.parse_args()

    ptm = args.model_name.split("/")[-1].split("-")[0]
    print("You are using the pretrain model: ", ptm)

    # if your model name contains the dataset name  `common_gen`, we will skip the warmup
    base_model_cont = WARM_UP_PATH + ptm
    if DATASET in args.model_name.lower():
        args.warmup = False
        base_model_cont = args.model_name

    inference_param = " --alpha 0.2 --max_length 70 --length_pen 2.0 "
    if args.mode != "train":
        test_cmd = f"python inference.py --gpus {args.gpus} --dataset {DATASET} " \
                   f" --batch_size {args.batch_size} --mode {args.mode} " \
                   f" --model_name {args.model_name} --save_path {args.save_path} --PTM {ptm} " \
                   f" --diversity_pen 0.0 --beam_size 4 --early_stop False {inference_param} "
        run(test_cmd)
    else:
        num_process = len(args.gpus.split(','))
        # distributed
        train_cmd = f"python train.py --gpus {args.gpus} --max_src_len 64 --max_tgt_len 64  --lr 1e-3 --dataset {DATASET} " \
                    f" --PTM {ptm} --mode train  --model_name {args.model_name} --accum_count {args.accum_count} " \
                    f" --diversity_pen 2.0 --beam_size 12  {inference_param} "
        if args.warmup:
            train_cmd += f" --warmup True --batch_size {args.warmup_batch_size} --n_epochs 10 --validate_every {args.validate_every} " \
                         f" --save_path {WARM_UP_PATH + ptm}   "
            run(train_cmd)

        train_cmd += f" --warmup False --batch_size {args.batch_size} --lr 2e-5 --n_epochs 1 " \
                     f" --validate_every {args.validate_every // 10} --reset_optimizer True --model_name {base_model_cont} " \
                     f" --save_path checkpoints/{DATASET}/{ptm} "
        run(train_cmd)
