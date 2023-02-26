import glob
import json
import os
import argparse


# import datasets

def load_jsonl(path):
    inst_list = []
    with open(path) as f:
        for line in f:
            inst_list.append(json.loads(line))
    return inst_list



def eval_function(sys_path):
    predictions = [0 for _ in range(len(sys_outputs))]
    for inst in sys_outputs:
        if isinstance(inst["sys_out"], list):
            sys_out = ' '.join(inst["sys_out"])
        else:
            sys_out = inst["sys_out"]
        assert isinstance(sys_out, str)
        position = int(inst["article_id"])
        predictions[position] = sys_out
    write_file = sys_path.replace(".sys", ".txt")
    with open(write_file, "w") as f:
        for pred in predictions:
            print(pred, file=f)
    sh_totto = f"bash language/totto/totto_eval.sh --prediction_path {write_file} --target_path jsonl_files/totto_meta/totto_dev_data.jsonl"
    print(sh_totto)
    os.system(sh_totto)



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
        sys_path = cand_file
        print("evaluate: ", sys_path)
        sys_outputs = load_jsonl(sys_path)
        eval_function(sys_path)
