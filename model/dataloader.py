from time import time
from datetime import timedelta
from fastNLP.io import JsonLoader
from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.pipe.pipe import Pipe


# set bos_id = pad_id
class Seq2SeqLoader(JsonLoader):
    def __init__(self, bos_id, pad_id, eos_id, max_src_len, max_tgt_len, source="src_id", target="tgt_id"):
        super(Seq2SeqLoader, self).__init__()
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.source = source
        self.target = target
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.bos_id = bos_id

    def _load(self, path):
        dataset = super(Seq2SeqLoader, self)._load(path)
        return dataset

    def load(self, paths):

        def get_tgt_inp(instance):
            return [self.pad_id] + instance[self.target][:-1]

        def get_tgt_outp(instance):
            return instance[self.target]

        def truncate_tgt_id(instance, max_len):

            if len(instance[self.target]) > max_len:
                return instance[self.target][:max_len - 1]
            else:
                return instance[self.target]

        def truncate_src_id(instance, max_len):
            if len(instance[self.source]) > max_len:
                return instance[self.source][:max_len - 1] + [self.eos_id]
            else:
                return instance[self.source]

        print('Start loading datasets !!!')
        start = time()

        # load datasets
        datasets = {}
        for name in paths:
            datasets[name] = self._load(paths[name])

            if name == 'train':
                datasets[name].apply(lambda ins: truncate_tgt_id(ins, self.max_tgt_len),
                                     new_field_name=self.target)
            datasets[name].apply(lambda ins: get_tgt_inp(ins),
                                 new_field_name='target_inp')
            datasets[name].apply(lambda ins: get_tgt_outp(ins),
                                 new_field_name='target_outp')
            datasets[name].apply(lambda ins: truncate_src_id(ins, self.max_src_len),
                                 new_field_name='src_inp')
            datasets[name].apply(lambda ins: len(ins["target_outp"]), new_field_name="seq_len")

            # drop some instance

            # set input and target
            datasets[name].set_input('src_inp', 'target_inp', "target_outp")
            datasets[name].set_pad_val('src_inp', self.pad_id)
            datasets[name].set_pad_val('target_inp', self.pad_id)
            datasets[name].set_pad_val('target_outp', self.pad_id)

        print('Finished in {}'.format(timedelta(seconds=time() - start)))

        return DataBundle(datasets=datasets)


class Seq2SeqPipe(Pipe):

    def __init__(self, args):
        super(Seq2SeqPipe, self).__init__()
        self.args = args

    def process(self, data_bundle):
        return data_bundle

    def process_from_file(self, paths):
        data_bundle = Seq2SeqLoader(self.args.bos_id, self.args.pad_id, self.args.eos_id, self.args.max_src_len,
                                    self.args.max_tgt_len).load(paths)
        return self.process(data_bundle)
