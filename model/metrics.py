from fastNLP import Metric
import itertools


class CoNTValidMetric(Metric):
    def __init__(self):
        super(CoNTValidMetric, self).__init__()
        self.torch_ngram = []

    # get validation loss
    def update(self, score):
        self.torch_ngram.append(score)

    def get_metric(self, reset=True):
        torch_ngrams = self.all_gather_object(self.torch_ngram)
        torch_ngram = list(itertools.chain(*torch_ngrams))
        eval_result = sum(torch_ngram) / len(torch_ngram)
        if reset:
            self.torch_ngram = []
        return {"torch_ngram": eval_result.item()}
