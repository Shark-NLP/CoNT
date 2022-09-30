from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase
from fastNLP import logger


class Loss(LossBase):
    def __init__(self):
        super(Loss, self).__init__()
        self._init_param_map()

    def get_loss(self, loss):
        return loss


class MLEValidMetric(MetricBase):
    def __init__(self):
        super(MLEValidMetric, self).__init__()
        self.valid_loss = []

    # get validation loss
    def evaluate(self, loss):
        self.valid_loss.append(-loss.mean())

    def get_metric(self, reset=True):
        eval_result = sum(self.valid_loss) / len(self.valid_loss)
        if reset:
            self.valid_loss = []
        return {"likelihood": eval_result.item()}


class CoNTValidMetric(MetricBase):
    def __init__(self):
        super(CoNTValidMetric, self).__init__()
        self.torch_ngram = []

    # get validation loss
    def evaluate(self, score):
        self.torch_ngram.append(score)

    def get_metric(self, reset=True):
        eval_result = sum(self.torch_ngram) / len(self.torch_ngram)
        if reset:
            self.torch_ngram = []
        return {"torch_ngram": eval_result.item()}
