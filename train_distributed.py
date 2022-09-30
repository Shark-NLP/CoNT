import glob
import sys

import torch
from transformers import AutoTokenizer

sys.path.append('../')
import argparse
import os
from os.path import exists
from model.dataloader import Seq2SeqPipe
from model.model import  CoNTGenerator
from model.metrics import Loss
from model.callback import MLECallback, CoNTCallback
from fastNLP import DistTrainer, get_local_rank
import torch.distributed as dist
from model.optimizer import Adafactor
from model.metrics import MLEValidMetric, CoNTValidMetric


def get_data_path(PTM, dataset):
    paths = {}
    paths['train'] = f'tokenized_files/{dataset}/train.{PTM}.jsonl'
    paths['val'] = f'tokenized_files/{dataset}/val.{PTM}.jsonl'
    return paths


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def configure_training(args):
    devices = [int(gpu) for gpu in args.gpus.split(',')]
    params = {}
    params['beam_size'] = args.beam_size
    params['batch_size'] = args.batch_size
    params['accum_count'] = args.accum_count
    params['margin'] = args.margin
    params['n_epochs'] = args.n_epochs
    params['validate_every'] = args.validate_every
    return devices, params


def train_model(args):
    # initialize
    dist.init_process_group(backend="nccl")
    if get_local_rank() != 0:
        dist.barrier()
    # load the datasets
    data_paths = get_data_path(args.PTM, args.dataset)
    if args.PTM == "codet5":
        tokenize_name = "Salesforce/codet5-base"
    elif args.PTM == "t5":
        tokenize_name = "t5-small"
    elif args.PTM == "pegasus":
        tokenize_name = "google/pegasus-xsum"
    else:
        raise NotImplementedError("please add this pretrained model ")
    tokenizer = AutoTokenizer.from_pretrained(tokenize_name)
    args.pad_id = tokenizer.pad_token_id
    args.eos_id = tokenizer.eos_token_id
    args.bos_id = tokenizer.bos_token_id
    model = CoNTGenerator(args.PTM, args.model_name, args.pad_id, args)

    if args.warmup:
        print("=" * 10, " Warmup with MLE ...", "=" * 10)
        callbacks_master = [MLECallback(args)]
        valid_metric = MLEValidMetric()
    else:
        print("=" * 10, "training contrastive-based model...", "=" * 10)
        callbacks_master = [CoNTCallback(args)]
        valid_metric = CoNTValidMetric()
    for name in data_paths:
        assert exists(data_paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)
    if get_local_rank() == 0:
        dist.barrier()

    devices, train_params = configure_training(args)
    optimizer = Adafactor(
        model.parameters(),
        lr=args.lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    if not args.reset_optimizer:
        optim_pt = torch.load(args.model_name + ".optm")
        optimizer.load_state_dict(optim_pt)
        print("=" * 20, "load optimizer from", args.model_name + ".optm")

    criterion = Loss()
    datasets = Seq2SeqPipe(args).process_from_file(data_paths)
    print(f'Information of {args.dataset}:', datasets)
    train_set = datasets.datasets['train']
    dev_set = datasets.datasets['val']
    trainer = DistTrainer(train_data=train_set, model=model, optimizer=optimizer,
                          loss=criterion, batch_size_per_gpu=args.batch_size,
                          update_every=args.accum_count, n_epochs=args.n_epochs, dev_data=dev_set,
                          print_every=10, validate_every=args.validate_every, metrics=valid_metric,
                          callbacks_master=callbacks_master)

    print('Start training with the following hyper-parameters:')
    print(train_params)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training/testing of CoNT'
    )

    parser.add_argument('--save_path', required=True,
                        help='root of the model', type=str)
    parser.add_argument('--gpus', default="0,1,2,3",
                        help='available gpus for training(separated by commas)', type=str)
    parser.add_argument('--batch_size', default=32,
                        help='the training batch size', type=int)
    parser.add_argument('--accum_count', default=1,
                        help='number of updates steps to accumulate before performing a backward/update pass', type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--margin', default=0.01,
                        help='parameter for MarginRankingLoss', type=float)
    parser.add_argument('--n_epochs', default=50,
                        help='total number of training epochs', type=int)
    parser.add_argument('--validate_every', default=2000,
                        help='number of update steps for validation and saving checkpoint', type=int)
    parser.add_argument('--max_sample_num', default=16, type=int)
    parser.add_argument('--n_gram', default=2, type=int)
    parser.add_argument('--dataset', default="wmt16")
    parser.add_argument('--warmup', type=str2bool)
    parser.add_argument('--PTM', default="t5")
    parser.add_argument('--reset_optimizer', type=str2bool, default=True)
    parser.add_argument('--scratch', type=str2bool, default=False)
    parser.add_argument('--model_name', default="google/pegasus-xsum")
    parser.add_argument('--max_src_len', default=512, type=int)
    parser.add_argument('--max_tgt_len', default=128, type=int)
    # inference parameters
    parser.add_argument('--min_length', default=5, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--beam_size', default=12, type=int)
    parser.add_argument('--early_stop', default=True, type=str2bool)
    parser.add_argument('--no_repeat_ngram', default=4, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--diversity_pen', default=1.0, type=float)
    parser.add_argument('--length_pen', default=2.0, type=float)

    # no need to set
    parser.add_argument('--pad_id', type=int)
    parser.add_argument('--eos_id', type=int)
    parser.add_argument('--bos_id', default=None, type=int)

    args = parser.parse_known_args()[0]
    train_model(args)
