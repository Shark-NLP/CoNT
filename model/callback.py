import os
import torch
from sortedcontainers import SortedList

from fastNLP.core.callback import Callback


class CoNTCallback(Callback):
    def __init__(self, args):
        super(CoNTCallback, self).__init__()
        self.args = args
        self.dev_results = SortedList([])
        self.cl_loss_list = []
        self.nll_loss_list = []
        self.patience = 3
        self.wait = 0

    def _save_this_model(self, model_path):
        try:
            torch.save(self.model, model_path)
            if len(self.dev_results) > self.patience:
                del_model = self.dev_results.pop(0)[1]
                os.remove(del_model)
            print(f" ============= save model at {model_path} ============= ")
        except Exception as e:
            print(f"The following exception:{e} happens when save {model_path}.")

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        score = eval_result["CoNTValidMetric"]["torch_ngram"]
        save_dir = os.path.join(self.args.save_path, self.trainer.start_time)
        name = "epoch-{}_step-{}.pt".format(self.epoch, self.step // self.args.accum_count)
        model_path = os.path.join(save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        if len(self.dev_results) < self.patience:
            self.dev_results.add((score, model_path))
            self._save_this_model(model_path)
        else:
            if score >= self.dev_results[0][0]:
                self.dev_results.add((score, model_path))
                self._save_this_model(model_path)
                self.wait = 0
                print("save hyperparams in ", os.path.join(save_dir, "hyperparams.txt"))
                with open(os.path.join(save_dir, "hyperparams.txt"), "w") as f:
                    for key, value in self.args.__dict__.items():
                        print(key, ":", value, file=f)
            else:
                if self.wait == self.patience:
                    print("early stop is triggered !!")
                    os.system("pkill -f train_distributed.py")
                else:
                    self.wait += 1

    def on_epoch_end(self):
        self.pbar.write('Epoch {} is done !!!'.format(self.epoch))

    def on_loss_begin(self, batch_y, predict_y):
        cl_loss = predict_y["cl_loss"].detach().cpu().item()
        nll_loss = predict_y["loss"].detach().cpu().item() - cl_loss
        self.cl_loss_list.append(cl_loss)
        self.nll_loss_list.append(nll_loss)
        if self.step // self.args.accum_count % (self.args.validate_every // 4) == 0:
            self.pbar.write(
                f'Contrastive loss is {sum(self.cl_loss_list) / len(self.cl_loss_list)}, nll_loss is {sum(self.nll_loss_list) / len(self.nll_loss_list)}')
            self.cl_loss_list = []
            self.nll_loss_list = []

    def on_exception(self, exception):
        if isinstance(exception, KeyboardInterrupt):
            os.system("pkill -f train_distributed.py")
        else:
            raise exception


class MLECallback(Callback):
    def __init__(self, args):
        super(MLECallback, self).__init__()
        self.args = args
        self.wait = 0
        self.patience = 3

    def _save_model(self):
        model = self.model
        save_dir = f"pretrained_weights/{self.args.dataset}"
        save_model = os.path.join(save_dir, self.args.PTM)
        os.makedirs(save_dir, exist_ok=True)
        optm_path = save_model + ".optm"

        model.generator.save_pretrained(save_model)
        torch.save(self.trainer.optimizer.state_dict(), optm_path)

        print(f" ============= save model at {save_model} ============= ")
        print(f" ============= save optimizer at {optm_path} ============= ")

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if not is_better_eval:
            # current result is getting worse
            if self.wait == self.patience:
                print("early stop is triggered !!")
                os.system("pkill -f train_distributed.py")
            else:
                self.wait += 1
        else:
            self._save_model()
            self.wait = 0

    def on_epoch_end(self):
        self.pbar.write('Epoch {} is done !!!'.format(self.epoch))

    def on_exception(self, exception):
        if isinstance(exception, KeyboardInterrupt):
            os.system("pkill -f train_distributed.py")
        else:
            raise exception
