# CoNT
This is the [transformers-based](https://github.com/huggingface/transformers.git) implementation 
 for NeurIPS 2022  paper: *[CoNT: Contrastive Neural Text Generation](https://arxiv.org/abs/2205.14690)*.
 For machine translation task please refer to our [fairseq code]().

CoNT is a Strong contrastive learning framework for neural text generation which outperforms the MLE based training method on **five** generation tasks, including *machine translation*, *summarization*, *code comment generation*, *data-to-text generation*, *commensense generation*. 
## Dependencies
Main libraries
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.7 +
- [transformers](https://github.com/huggingface/transformers) 4.21.0
- [fastNLP](https://github.com/fastnlp/fastNLP) 1.0.0beta
```
pip install -r requirements.txt
```

	
All code only supports running on Linux.


### data
we have prepared the raw dataset to help you reproduce the results in our paper.  Datasets provided by this repo can  **only**  be used for *Reproduction* and *Research Purposes*.
All files are in `jsonl` format where each line is a `json` sample:
```
{"source": "the input text (for encoder)", "target": "the target text (for decoder)"}
```
You can Download the jsonl files through these links.
1. Summarization：
    - [XSum](https://drive.google.com/file/d/1t--UZo4Pnv4HjGhAfun5vDz3JCoqIggq/view?usp=sharing)
    - [Multi-News](https://drive.google.com/file/d/16VdfzvLmmOrYsayujA-Hu4d3i_ejHTln/view?usp=sharing)
2. Translation：
    - [WMT'16 Ro-En](https://drive.google.com/file/d/1rGoylmZvIhNvsoPZda7OZP_0nYUfUpoq/view?usp=sharing)
    - WMT'14 En-De and IWSLT'14 De-En (fairseq folder)
3. Code comment Generation
    - [java](https://drive.google.com/file/d/1PBdxKvMTvfCzseactMRffTUwuTI7oAGz/view?usp=sharing)
    - [python](https://drive.google.com/file/d/189xlRW3r3UuMTko73zURfJ3I_LXQ026D/view?usp=sharing)
4. Data-to-text generation  
    - [Wikibio](https://drive.google.com/file/d/1i0BZykxifH2hEdCyB_nZFvs2PT4UdUFJ/view?usp=sharing)
    - [ToTTo](https://drive.google.com/file/d/1nOlhGKpTWPCmAwmEI_gdALkAXlMn2Tbk/view?usp=sharing) (Blind test set)
5. Commonsense generation  
    - [CommonGen](https://drive.google.com/file/d/1UvCBenGMzdQyR25ka_1vmaPwGVFQzqvS/view?usp=sharing) (Blind test set)
Before loading the training set, please pre-tokenize these files  with the following command:
```
mkdir jsonl_files
mkdir tokenized_files
mv /download/path/xsum.zip  ./jsonl_files
cd jsonl_files
unzip xsum.zip && cd ..
python preprocess/preprocess.py --model_name  t5-small --dataset xsum
``` 
This command will produce the tokenized files of XSum `tokenized_files/train.t5.jsonl, tokenized_files/val.t5.jsonl` with the tokenizer of t5-small  

### Training
We have provided the training script for each datasets, and you can easily start the training process with them:
```#If you do not have a warmed-up checkpoint, you should use --warmup True to train the generation model with NLLLoss 
python run_xsum.py --mode train --gpus 0,1,2,3 --warmup True --model_name t5-small (or google/pegasus-large, google/pegasus-xsum)
```
the warmed-up checkpoint will be saved to `./pretrained_weigths/xsum/t5(or pegasus)` by default.  
Please notice that huggingface also provides many finetuned checkpoints. So that if the `--model_name`  contains the dataset name  (like `google/pegasus-xsum`), we will skip the warmup.
```
#If you already have a warmed-up checkpoint, you can skip the warm-up stage
python run_xsum.py --mode train --gpus 0,1,2,3 --warmup False
```

After completing the training process, several best checkpoints are stored in a folder named after the training start time, for example, `./checkpoints/xsum/t5/2022-10-05-10-37-24-196200`

### Generation
You can run the following command to generate the results on test/validation set with all checkpoints in a given folder:
You can first select the best checkpoint on dev set with `--mode val` and then generate the results on the test set  with that checkpoint. 

```
python run_xsum.py --mode test/val --model_name t5-small --save_path checkpoints/xsum/t5/2022-10-05-10-37-24-196200/ --gpus 0,1,2,3,4,5,6,7
```
This will produce the generated results in: `results/xsum/t5/2022-10-05-10-37-24-196200/` -- `epoch-2_step-8000.val.sys` , `epoch-2_step-8000.val.ref`, `epoch-2_step-10000.val.sys` , `epoch-2_step-10000.val.ref`


To generate the results for test set with  **a specified checkpoint**  using the `--ckpt ` parameter and change the mode to `test`:
```
python run_xsum.py --mode test --model_name t5-small --save_path checkpoints/xsum/t5/2022-10-05-10-37-24-196200/ --ckpt epoch-2_step-8000.pt --gpus 0,1,2,3,4,5,6,7
```
This will produce the generated results in: `results/xsum/t5/2022-10-05-10-37-24-196200/` -- `epoch-2_step-8000.test.sys` , `epoch-2_step-8000.test.ref`

### Evaluation
This is an example to evaluate all the generated results in this folder:
```
python evaluation/xsum.py --sys_path results/xsum/t5/2022-10-05-10-37-24-196200/
```
If you only want to evaluate a specified file：
```
python evaluation/xsum.py --sys_file results/xsum/t5/2022-10-05-10-37-24-196200/epoch-2_step-8000.sys
```

### Details about evaluation scripts and preprocess
Comming soon

-----
**We are pleased to answer any questions about this paper or codes** ^_^
email: `cxan20@fudan.edu.cn` 
  WeChat: `cxan996`

### Citing
```
@article{an2022cont,
  title={CoNT: Contrastive Neural Text Generation},
  author={An, Chenxin and Feng, Jiangtao and Lv, Kai and Kong, Lingpeng and Qiu, Xipeng and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2205.14690},
  year={2022}
}
```
