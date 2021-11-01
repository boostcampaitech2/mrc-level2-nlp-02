import sys
sys.path.append("/opt/ml/code/")

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

import model_encoder

dataset = load_from_disk('../data/train_dataset')['validation']

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    
set_seed(42) # magic number :)

class DenseRetrieval:
  def __init__(self, args, train_dataset, valid_dataset, num_neg, tokenizer, p_encoder, q_encoder):

    self.args = args
    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset
    self.num_neg = num_neg

    self.tokenizer = tokenizer
    self.p_encoder = p_encoder
    self.q_encoder = q_encoder
    
    self.train_answer_index = []
    self.valid_answer_index = []
    
    self.prepare_in_batch_negative(train_dataset, num_neg, tokenizer, is_train=True)
    self.prepare_in_batch_negative(valid_dataset, num_neg, tokenizer, is_train=False)
    
    

  def prepare_in_batch_negative(self, dataset, num_neg, tokenizer, is_train):
    
    corpus = np.array(list(set([example for example in dataset['context']])))
    p_with_neg = []

    for c, q in zip(dataset['context'], dataset['question']):
      if is_train:
        p_with_neg.append(c)
        neg_count = 0

        responses = es.search(index='wiki_documents', size=num_neg*2, query={'match':{'text':q}})
        for res in responses['hits']['hits']:
          if c != res['_source']['text'] and neg_count < num_neg:
            neg_count += 1
            p_with_neg.append(res['_source']['text'])
          if not neg_count < num_neg:
            break
        while neg_count < num_neg:
          neg_idxs = np.random.randint(len(corpus), size=num_neg-neg_count)

          if c not in corpus[neg_idxs]:
            p_neg = corpus[neg_idxs].tolist()

            p_with_neg.extend(p_neg)
            break
        # breakpoint()
        # 정답 섞기
        # number_for_swap = random.randint(1, num_neg)
        # p_with_neg[-(num_neg+1)], p_with_neg[-number_for_swap] = p_with_neg[-number_for_swap], p_with_neg[-(num_neg+1)]
        # self.train_answer_index.append(num_neg+1-number_for_swap)
    
      else:
        while True:
          neg_idxs = np.random.randint(len(corpus), size=num_neg)

          if c not in corpus[neg_idxs]:
            p_neg = corpus[neg_idxs].tolist()

            p_with_neg.append(c)
            p_with_neg.extend(p_neg)
            # print(p_with_neg)
            break
            
        # number_for_swap = random.randint(1, num_neg+1)
        # p_with_neg[-(num_neg+1)], p_with_neg[-number_for_swap] = p_with_neg[-number_for_swap], p_with_neg[-(num_neg+1)]
        # self.valid_answer_index.append(num_neg+1-number_for_swap)
        
        
        

    q_seqs = tokenizer(
      dataset['question'],
      padding = 'max_length',
      truncation = True,
      return_tensors = 'pt',
      max_length = 510,
      # return_token_type_ids=False,
    )

    p_seqs = tokenizer(
      p_with_neg,
      padding = 'max_length',
      truncation=True,
      return_tensors = 'pt',
      max_length = 510,
      # return_token_type_ids=False,
    )


    max_len = p_seqs['input_ids'].size(-1)
    p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
    p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
    # p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

 

    if is_train:
      train_dataset = TensorDataset(
        p_seqs['input_ids'], p_seqs['attention_mask'], #p_seqs['token_type_ids'],
        q_seqs['input_ids'], q_seqs['attention_mask'], #q_seqs['token_type_ids'],
        
      )
      self.train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size = self.args.per_device_train_batch_size
      )
    else:
      valid_dataset = TensorDataset(
        p_seqs['input_ids'], p_seqs['attention_mask'], #p_seqs['token_type_ids'],
        q_seqs['input_ids'], q_seqs['attention_mask'], #q_seqs['token_type_ids'],
      )
      self.valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size = self.args.per_device_train_batch_size
      )

    valid_seqs = tokenizer(
      dataset['context'],
      padding='max_length',
      truncation=True,
      return_tensors = 'pt',
      max_length = 510,
    )

    passage_dataset = TensorDataset(
      valid_seqs['input_ids'],
      valid_seqs['attention_mask'],
      #valid_seqs['token_type_ids'],
    )
    
    if is_train:
      self.passage_dataloader = DataLoader(
        passage_dataset,
        batch_size = self.args.per_device_train_batch_size
      )
    

  def train(self, args=None):
    if args is None:
      args = self.args
    batch_size = args.per_device_train_batch_size

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
      {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
      {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
      {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(
      optimizer_grouped_parameters,
      lr=args.learning_rate,
      eps=args.adam_epsilon
    )
    t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps = args.warmup_steps,
      num_training_steps = t_total
    )

    global_step = 0

    self.p_encoder.zero_grad()
    self.q_encoder.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = tqdm(range(int(args.num_train_epochs)), desc='Epoch')

    min_loss = 9999999
    for _ in train_iterator:
      valid_check_period = 100
      count_iteration = 0
      total_loss = 0
      train_total = 0
      train_correct = 0
      print('train_iterator')

      with tqdm(self.train_dataloader, unit='batch') as tepoch:
        print('with tqdm')
        for i, batch in enumerate(tepoch):
          # print('batch')
          self.p_encoder.train()
          self.q_encoder.train()
          targets = torch.zeros(batch_size).long()
          # targets = torch.tensor(self.train_answer_index[i*batch_size : i*batch_size+batch_size])
          targets = targets.to(args.device)
          

          p_inputs = {
            "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
            "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
           # "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device),
          }


          q_inputs = {
            "input_ids": batch[2].to(args.device),
            "attention_mask": batch[3].to(args.device),
            #"token_type_ids" : batch[5].to(args.device),
          }

          p_outputs = self.p_encoder(**p_inputs)
          q_outputs = self.q_encoder(**q_inputs)
          
          # print(p_outputs.shape, q_outputs.shape)

          # 기존의것 p_outputs = p_outputs.view(batch_size, -1, self.num_neg+1)
          # 내가시도 q_outputs = torch.transpose(q_outputs.view(batch_size, -1, 1),1,2)
            
          # 틀림 q_outputs = q_outputs.view(batch_size, 1, -1)
        
          p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1).transpose(1, 2)
          q_outputs = q_outputs.view(batch_size, 1, -1)
          
          # print(p_outputs.shape, q_outputs.shape)

          sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
          # breakpoint()
          sim_scores = sim_scores.view(batch_size, -1)
          sim_scores = F.log_softmax(sim_scores, dim=1)

          train_total += targets.size(0)
          train_correct += torch.argmax(sim_scores, dim=1).eq(targets).sum().item()

          loss = F.nll_loss(sim_scores, targets)
          tepoch.set_postfix(loss=f'{str(loss.item())}')

          loss.backward()
          optimizer.step()
          scheduler.step()



          self.p_encoder.zero_grad()
          self.q_encoder.zero_grad()

          global_step += 1

          torch.cuda.empty_cache()

          del p_inputs, q_inputs

          # print('loss:', loss)
          total_loss += loss
          count_iteration += 1
          # print(count_iteration)

          if count_iteration == valid_check_period:
            count_iteration = 0
            print('total loss: ', total_loss)
            print('train accuracy: ', train_correct/train_total)
            total_loss = 0
            train_total = 0
            train_correct = 0

            valid_loss = 0
            valid_total = 0
            valid_correct = 0

            with torch.no_grad():
              self.p_encoder.eval()
              self.q_encoder.eval()

              with tqdm(self.valid_dataloader, unit='batch') as tepoch_val:
          
                for j, batch_val in enumerate(tepoch_val):
                  targets = torch.zeros(batch_size).long()
                  # targets = torch.tensor(self.valid_answer_index[j*batch_size : j*batch_size+batch_size])
                  targets = targets.to(args.device)
                  

                  p_inputs = {
                    "input_ids": batch_val[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                    "attention_mask": batch_val[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                    #"token_type_ids": batch_val[2].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                  }


                  q_inputs = {
                    "input_ids": batch_val[2].to(args.device),
                    "attention_mask": batch_val[3].to(args.device),
                    #"token_type_ids": batch_val[5].to(args.device),
                  }

                  p_outputs = self.p_encoder(**p_inputs)
                  q_outputs = self.q_encoder(**q_inputs)
                  # print('val: ',p_outputs.shape, q_outputs.shape)
                
                  # p_outputs = p_outputs.view(batch_size, -1, self.num_neg+1)
                  # q_outputs = torch.transpose(q_outputs.view(batch_size, -1, 1),1,2)
                  # q_outputs = q_outputs.view(batch_size, 1, -1)
                  
                
                  p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1).transpose(1, 2)
                  q_outputs = q_outputs.view(batch_size, 1, -1)
                  # print('val: ',p_outputs.shape, q_outputs.shape)
                  
                  sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
                  # breakpoint()
                  sim_scores = sim_scores.view(batch_size, -1)
                  sim_scores = F.log_softmax(sim_scores, dim=1)

                  valid_total += targets.size(0)
                  valid_correct += torch.argmax(sim_scores, dim=1).eq(targets).sum().item()

                  loss = F.nll_loss(sim_scores, targets)
                  tepoch_val.set_postfix(loss=f'{str(loss.item())}')

                  torch.cuda.empty_cache()

                  del p_inputs, q_inputs

                  valid_loss += loss
              
              print('valid_loss: ', valid_loss)
              print('min_loss: ', min_loss)
              print('accuracy: ', valid_correct/valid_total)
              if valid_loss < min_loss:
                print('New min loss, so saving the model.')
                retriever.p_encoder.save_pretrained('encoders/p_encoder_neg_sen')
                retriever.q_encoder.save_pretrained('encoders/q_encoder_neg_sen')
                min_loss = valid_loss



  def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):
    if args is None:
      args = self.args
    
    if p_encoder is None:
      p_encoder = self.p_encoder
    
    if q_encoder is None:
      q_encoder = self.q_encoder

    with torch.no_grad():
      p_encoder.eval()
      q_encoder.eval()

      q_seqs_val = self.tokenizer(
        [query],
        padding = 'max_length',
        truncation= True,
        return_tensors = 'pt',
        # return_token_type_ids=False,
      ).to(args.device)
      q_emb = q_encoder(**q_seqs_val).to('cpu')

      p_embs = []
      for batch in self.passage_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        p_inputs = {
          'input_ids': batch[0],
          'attention_mask': batch[1],
          'token_type_ids': batch[2]
        }
        p_emb = p_encoder(**p_inputs).to('cpu')
        p_embs.append(p_emb)
    
    p_embs = torch.stack(
      p_embs, dim = 0
    ).view(len(self.passage_dataloader.dataset), -1)

    dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

    return rank[:k]

class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    self.bert = BertModel(config)
    self.init_weights()

  def forward(self, input_ids, attention_mask=None, token_type_ids=None):
    outputs = self.bert(
      input_ids,
      attention_mask = attention_mask,
      token_type_ids = token_type_ids
    )

    pooled_output = outputs[1]
    return pooled_output

train_dataset = load_from_disk('../data/train_dataset')['train']
valid_dataset = load_from_disk('../data/train_dataset')['validation']

args = TrainingArguments(
  output_dir = 'dense_retrieval',
  evaluation_strategy = 'epoch',
  # learning_rate=1e-5,
  # learning_rate=3e-4,
  # learning_rate=5e-5,
  learning_rate=5e-6,
  # learning_rate=1e-7,
  per_device_train_batch_size=2,
  per_device_eval_batch_size=2,
  num_train_epochs=30,
  weight_decay=0.01
)

# model_checkpoint = 'klue/bert-base'

model_checkpoint = 'Huffon/sentence-klue-roberta-base'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
# q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)

p_encoder = model_encoder.RobertaEncoder.from_pretrained(model_checkpoint).to(args.device)
q_encoder = model_encoder.RobertaEncoder.from_pretrained(model_checkpoint).to(args.device)

retriever = DenseRetrieval(
  args=args,
  train_dataset=train_dataset,
  valid_dataset=valid_dataset,
  # num_neg=12,
  num_neg=15,
  tokenizer=tokenizer,
  p_encoder=p_encoder,
  q_encoder=q_encoder
)

retriever.train()

# retriever.p_encoder.save_pretrained('encoders/p_encoder_neg_sen')
# retriever.q_encoder.save_pretrained('encoders/q_encoder_neg_sen')