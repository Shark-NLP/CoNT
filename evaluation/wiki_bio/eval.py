import glob
import json
import os
import time
from multiprocessing import Pool

import shutil
import sys
import codecs
import nltk
import logging

from pyrouge import Rouge155
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import argparse


# import datasets


def load_jsonl(path):
    inst_list = []
    with open(path) as f:
        for line in f:
            inst_list.append(json.loads(line))
    return inst_list


def process(data, tmp='tmp_pyrouge'):
    # tmp='/tmp/tmp_pyrouge'
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = str(time.time()).replace('.', '')
    tmp_dir = tmp + "/{}{}".format(current_time, pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")

    def write(url, s):
        with open(url, 'w', encoding='utf-8') as f:
            f.write(s)

    for i in range(cnt):
        if len(references[i]) < 1:
            continue

        write(tmp_dir + "/candidate/cand.{}.txt".format(i), candidates[i])
        write(tmp_dir + "/reference/ref.{}.txt".format(i), references[i])

    r = Rouge155()
    r.log.setLevel(logging.WARN)
    r.model_dir = tmp_dir + "/reference/"
    r.system_dir = tmp_dir + "/candidate/"
    r.model_filename_pattern = 'ref.#ID#.txt'
    r.system_filename_pattern = r'cand.(\d+).txt'
    rouge_results = r.convert_and_evaluate()

    results_dict = r.output_to_dict(rouge_results)

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return results_dict


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def test_rouge(cand, ref, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)
    candidates_chunks = list(chunks(candidates, int(len(candidates) / num_processes)))
    references_chunks = list(chunks(references, int(len(references) / num_processes)))
    n_pool = len(candidates_chunks)
    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i], references_chunks[i], i))
    pool = Pool(n_pool)
    results = pool.map(process, arg_lst)
    final_results = {}
    for i, r in enumerate(results):
        for k in r:
            if (k not in final_results):
                final_results[k] = r[k] * len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k] / len(candidates)
    return final_results


# preds are lists consists of summaries.
# each summary is a string: sentence1, \n , sentence2, \n
def pyrouge_score(preds, labels, num_processes=16):
    tmp = "tmp_pyrouge"
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    def split_sent(s):
        return '\n'.join(nltk.sent_tokenize(s))

    preds = list(map(split_sent, preds))
    labels = list(map(split_sent, labels))

    results = test_rouge(cand=preds, ref=labels, num_processes=num_processes)
    res_dict = {'rouge4': results['rouge_4_f_score'] * 100}
    print(res_dict)
    return res_dict


def get_rouge_score():
    predictions = []
    golds = []
    for inst in sys_outputs:
        if isinstance(inst["sys_out"], list):
            sys_out = ' '.join(inst["sys_out"])
        else:
            sys_out = inst["sys_out"]
        gold = ref_dict[inst["article_id"]].replace("\n", ' ')
        assert isinstance(sys_out, str)
        assert isinstance(gold, str)
        predictions.append(sys_out)
        golds.append(gold)

    ret_dict = pyrouge_score(predictions, golds)
    print(ret_dict)


def get_bleu_score():
    predictions = []
    golds = []
    for inst in sys_outputs:
        if isinstance(inst["sys_out"], list):
            sys_out = ' '.join(inst["sys_out"])
        else:
            sys_out = inst["sys_out"]
        gold = ref_dict[inst["article_id"]].replace("\n", ' ')
        assert isinstance(sys_out, str)
        assert isinstance(gold, str)
        predictions.append(sys_out.split())
        golds.append([gold.split()])

    bleu = corpus_bleu(golds, predictions)
    print("bleu: ", bleu)


def build_tgt_dict(insts):
    test_dict = {}
    # test_dir = os.path.join("datasets", dataset, "test.id.jsonl")
    for inst in insts:
        gold = inst["ref_out"]
        if isinstance(gold, list):
            gold = ' '.join(gold)
        test_dict[inst["article_id"]] = gold
    return test_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys_file', default=None, type=str)
    parser.add_argument('--sys_path', default=None, type=str)
    args = parser.parse_args()
    if args.sys_path is not None:
        candidate_files = glob.glob(os.path.join(args.sys_path, "*.sys"))
        for candidate_file in candidate_files:
            sys_files = os.path.join(args.sys_path, candidate_file)
    else:
        candidate_files = [args.sys_file]
    for cand_file in candidate_files:
        ref_path = cand_file.replace('sys', 'ref')
        sys_path = cand_file
        print("evaluate: ", sys_path)
        sys_outputs = load_jsonl(sys_path)
        ref_outputs = load_jsonl(ref_path)
        ref_dict = build_tgt_dict(ref_outputs)
        get_rouge_score()
        get_bleu_score()
