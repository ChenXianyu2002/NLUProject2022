import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.activations import gelu


class PairSimNet(nn.Module):
    def __init__(self, args, model_name='bert-base-uncased'):
        super(PairSimNet, self).__init__()
        self.args = args
        self.model_name = model_name

        self.encoder = AutoModel.from_pretrained(model_name)
        # self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.hidden_size = 768
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.cls_head = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        cls_encoding = self.encoder(input_ids, attention_mask, token_type_ids).pooler_output
        enc = self.dropout(gelu(self.fc1(cls_encoding)))
        enc = self.cls_head(enc)
        return F.sigmoid(enc)


class PairSimNet2(PairSimNet):
    def __init__(self, args, model_name='bert-base-uncased'):
        super(PairSimNet2, self).__init__(args, model_name)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, sent1_input_ids, sent2_input_ids, sent1_attention_mask, sent2_attention_mask,
                sent1_token_type_ids=None, sent2_token_type_ids=None, **kwargs):
        sent1_enc = self.encoder(sent1_input_ids, sent1_attention_mask, sent1_token_type_ids).pooler_output
        sent2_enc = self.encoder(sent2_input_ids, sent2_attention_mask, sent2_token_type_ids).pooler_output

        fuse_enc = torch.cat((sent1_enc, sent2_enc), dim=1)
        enc = self.dropout(gelu(self.fc1(fuse_enc)))
        enc = self.cls_head(enc)
        return F.sigmoid(enc)
