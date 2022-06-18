from abc import ABC

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MyDataset(Dataset, ABC):
    def __init__(self, args, data_path, tokenizer=None):
        self.args = args
        self.cls_method = args.cls_method
        self.model_name = args.model_name
        self.data_path = data_path
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data = self.load_csv()

    def load_csv(self):
        df = pd.read_csv(self.data_path, sep='\t')
        useful_columns = ['Id', 'Question1', 'Question2']
        if 'Label' in df.columns:
            useful_columns.append('Label')
        data = list(zip(*[df[_].tolist() for _ in useful_columns]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_item = self.data[item]
        eid, sent1, sent2 = [data_item[_] for _ in range(3)]
        label = None
        if len(data_item) > 3:
            label = data_item[3]
        if self.cls_method == 'one-pass':
            encoded_sent = self.tokenizer(
                sent1, sent2,
                padding='max_length',
                truncation=True,
                max_length=self.args.max_input_length,
                return_tensors='pt'
            )
            if label is None:  # test
                return eid, encoded_sent
            else:  # train/dev
                return eid, encoded_sent, label
        elif self.cls_method == 'two-pass':
            encoded_sent1 = self.tokenizer(
                sent1, padding='max_length', truncation=True, max_length=self.args.max_input_length, return_tensors='pt'
            )
            encoded_sent2 = self.tokenizer(
                sent2, padding='max_length', truncation=True, max_length=self.args.max_input_length, return_tensors='pt'
            )
            if label is None:  # test
                return eid, encoded_sent1, encoded_sent2
            else:  # train/dev
                return eid, encoded_sent1, encoded_sent2, label
        else:
            raise ValueError(f'cls_method {self.cls_method} is not supported.')
