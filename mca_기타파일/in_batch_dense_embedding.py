import warnings
warnings.filterwarnings('ignore')

import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
    RobertaModel, RobertaPreTrainedModel

)
import transformers
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

training_dataset = load_from_disk('../data/train_dataset')['train']

from transformers import AutoTokenizer
import numpy as np

# model_checkpoint = "bert-base-multilingual-cased"
# model_checkpoint = "klue/roberta-base"
model_checkpoint = "klue/bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup

torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)



from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

# q_seqs = tokenizer(training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt', return_token_type_ids=False)
# p_seqs = tokenizer(training_dataset['context'], padding="max_length", truncation=True, return_tensors='pt', return_token_type_ids=False)
q_seqs = tokenizer(training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
p_seqs = tokenizer(training_dataset['context'], padding="max_length", truncation=True, return_tensors='pt')

train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
                        q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'],)

# class BertEncoder(RobertaPreTrainedModel):
class BertEncoder(BertPreTrainedModel):  
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    # self.bert = RobertaModel(config)
    self.bert = BertModel(config)
    self.init_weights()
      
  def forward(self, input_ids, 
              attention_mask=None, token_type_ids=None): 
  
      outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids #roberta시 주석
                          )
      
      pooled_output = outputs[1]

      return pooled_output


def train(args, dataset, p_model, q_model):
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
    {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
    {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
  ]

  optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=args.learning_rate,
    eps=args.adam_epsilon
  )
  
  # Dataloader
  train_sampler = RandomSampler(dataset)
  train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

# tt
  t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

  # Start training!
  global_step = 0
  
  p_model.zero_grad()
  q_model.zero_grad()
  torch.cuda.empty_cache()
  
  train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")

    for step, batch in enumerate(epoch_iterator):
      q_encoder.train()
      p_encoder.train()
      
      if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)

      p_inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] # roberta시 주석
                  }
      
      q_inputs = {'input_ids': batch[3],
                  'attention_mask': batch[4],
                  'token_type_ids': batch[5] # roberta시 주석
                  }
      
      p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
      q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)

      breakpoint()
      # Calculate similarity score & loss
      sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
      
      # print('q_outputs: ',q_outputs)
      # print('p_outputs: ',p_outputs)
      # print('sim_scores: ',sim_scores)
        
      # target: position of positive samples = diagonal element 
      targets = torch.arange(0, args.per_device_train_batch_size).long()
      if torch.cuda.is_available():
        targets = targets.to('cuda')

      sim_scores = F.log_softmax(sim_scores, dim=1)
      
      

      loss = F.nll_loss(sim_scores, targets)
      print(loss)

      loss.backward()
      optimizer.step()
      scheduler.step()
      q_model.zero_grad()
      p_model.zero_grad()
      global_step += 1
      
      torch.cuda.empty_cache()


    
  return p_model, q_model

args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01
)

p_encoder = BertEncoder.from_pretrained(model_checkpoint)
q_encoder = BertEncoder.from_pretrained(model_checkpoint)

if torch.cuda.is_available():
  p_encoder.cuda()
  q_encoder.cuda()


p_encoder, q_encoder = tqdm(train(args, train_dataset, p_encoder, q_encoder))