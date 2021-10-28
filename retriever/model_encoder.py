import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    RobertaModel,
    RobertaPreTrainedModel,
    PreTrainedModel,
    BertModel,
    BertPreTrainedModel,
)


class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        pooled_outputs = outputs[1]
        return pooled_outputs


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_outputs = outputs[1]
        return pooled_outputs