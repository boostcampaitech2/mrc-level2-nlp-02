import torch.nn as nn

from transformers import (
    AutoModel,
    AutoConfig,
    RobertaModel,
    RobertaPreTrainedModel,
    BertModel,
    BertPreTrainedModel,
)


class klueRobertaEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]

        cls_embedding = last_hidden_state[:, 0, :]
        pooled_output = self.dense(self.dropout(cls_embedding))
        pooled_output = self.activation(self.dropout(pooled_output))
        pooled_output = self.dropout(pooled_output)

        return pooled_output


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
