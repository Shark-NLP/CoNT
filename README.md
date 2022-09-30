# CoNT
This is the [transformers-based](https://github.com/huggingface/transformers.git) implementation for NeurIPS 2022  paper: *[CoNT: Contrastive Neural Text Generation](https://arxiv.org/abs/2205.14690)* 
CoNT is a Strong contrastive learning framework for neural text generation which outperforms the MLE based training method on five generation tasks. 

For machine translation task please refer to our fairseq code.

## Dependencies
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.7 +
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3
	- You should install pyrouge package first to reproduce our results. Instruction for installing pyrouge can be found in this [repo](https://github.com/ChenxinAn-fdu/CGSum)
- [rouge](https://github.com/pltrdy/rouge) 1.0.0
	- Used in  the validation phase.
- [transformers](https://github.com/huggingface/transformers) 4.20 + (for abstractive verison)

	
All code only supports running on Linux.

### we are active on opening source code and the full instruction of running our code will be available before 5th Oct
