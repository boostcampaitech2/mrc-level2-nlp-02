import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.file_utils import ModelOutput

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaQAWeightedLayerPooling(RobertaPreTrainedModel):
    """[summary]
    Encoder 상위 layer부터 hidden state를 weight pooling 결과 사용
    Args:
        layer_start [int] :
            pooling할 layer의 시작 값을 설정하는 값입니다.
            (layer_start는 음수 값으로 뒤에서 부터 호출,
             'all_hidden_states[self.layer_start:, :, :, :]')
        
        layer_weights [tensor table]:
            weight pooling에서 사용할 weight를 tensor형태로 전달합니다.
            None이 전달되면 모든 가중치는 1로 곱해집니다.

    """
    def __init__(self, model_name, config, layer_start: int = -4, layer_weights = None):
        super(RobertaQAWeightedLayerPooling, self).__init__(config)
        config.update({'output_hidden_states':True})
        self.config = config
        self.num_labels = config.num_labels
        
        self.roberta = RobertaModel.from_pretrained(model_name, config = config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.layer_start = layer_start
        num_hidden_layers = self.roberta.config.num_hidden_layers
        if layer_weights is not None :
            self.layer_weights = layer_weights
        else :
            # self.layer_weights = nn.Parameter(torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float))
            self.layer_weights = nn.Parameter(torch.tensor([1] * (-layer_start), dtype=torch.float))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # output[0] = last hidden state, output[1] = pooler_output (if 'add_pooling_layer=False' => 없어짐)
        # output[1] = all hidden states (if 'output_hidden_states': True) == past_key_value, tuple 형태로 반환
        
        all_hidden_states = torch.stack(outputs[1])
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
    
        # breakpoint()
        # sequence_output = outputs[0]
        # logits = self.qa_outputs(sequence_output)
        
        logits = self.qa_outputs(weighted_average)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) #+ outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            #hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaQALSTM(RobertaPreTrainedModel):
    def __init__(self, model_name, config, layer_start: int = -4, hidden_dim: int = 128, layer_weights = None):
        super(RobertaQALSTM, self).__init__(config)
        self.config = config
        self.hidden_dim = config.hidden_size ## 이거 확인
        self.num_labels = config.num_labels
        
        self.roberta = RobertaModel.from_pretrained(model_name, config = config, add_pooling_layer=False)

        # self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2, num_layers = 1, bidirectional = True, batch_first = True)
        # self.lstm = nn.LSTM(input_size = 1024, hidden_size = 1024, num_layers = 3, dropout=0.5, bidirectional = True, batch_first = True)
        # self.dense_layer = nn.Linear(2048, 30, bias=True)

        self.lstm = nn.LSTM(input_size =self.hidden_dim , hidden_size = self.hidden_dim , num_layers = 3, dropout=0.5, bidirectional = True, batch_first = True)
        self.qa_outputs = nn.Linear(self.hidden_dim*2, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # output[0] = last hidden state
        sequence_output = outputs[0]

        # sequence_output = outputs[0]
        # logits = self.qa_outputs(sequence_output)

        enc_hiddens, (last_hidden, last_cell) = self.lstm(sequence_output)
        logits = self.qa_outputs(enc_hiddens)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )