import elasticsearch
from elasticsearch import Elasticsearch
import pandas as pd
from datasets import DatasetDict, Dataset
import pororo
import torch
from tqdm import tqdm
from transformers import TrainingArguments, AutoModel, AutoTokenizer

from typing import List, Optional

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

from model_encoder import BertEncoder, RobertaEncoder


def create_elastic_object() -> Elasticsearch:
  """Elasticsearch와 연결하는 object 생성

  Returns:
      Elasticsearch: Elasticsearch Object
  """
  es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

  if es.ping():
    print('Elasticsearch object created.')
  else:
    print('Failed to make Elasticsearch object')
  return es

def run_elastic_sparse_retrieval(
  datasets: DatasetDict,
  training_args: TrainingArguments,
  data_args: DataTrainingArguments,
  ) -> DatasetDict:

  es = create_elastic_object()

  questions = datasets['validation']['question']
  ids = datasets['validation']['id']
  answers = datasets['validation']['answers']
  # breakpoint()
  relevent_contexts = []
  if data_args.use_entity_enrichment:
    print('Use entitiy enrichment.')

  for question in tqdm(questions):
    relevent_context = search_with_elastic(es, question, data_args)
    relevent_contexts.append(relevent_context)

  df = pd.DataFrame({'id':ids, 'question':questions, 'context':relevent_contexts, 'answers':answers})
  datasets = DatasetDict({"validation": Dataset.from_pandas(df)})

  return datasets

def run_elastic_dense_retrieval(
  datasets: DatasetDict,
  training_args: TrainingArguments,
  data_args: DataTrainingArguments,
  ) -> DatasetDict:

  es = create_elastic_object()

  questions = datasets['validation']['question']
  ids = datasets['validation']['id']
  if training_args.do_eval:
    answers = datasets['validation']['answers']########EVAL########
  
  if training_args.do_train:
    ids = datasets['train']['id']
    questions = datasets['train']['question']
    answers = datasets['train']['answers']
    contexts = datasets['train']['context']

  # breakpoint()
  # q_encoder = BertEncoder.from_pretrained('encoders/q_encoder_neg').to('cuda') ###########
  q_encoder_sen = RobertaEncoder.from_pretrained('encoders/q_encoder_neg_sen').to('cuda') 
  q_encoder_bert = BertEncoder.from_pretrained('encoders/q_encoder').to('cuda') 

  q_encoder = [q_encoder_sen, q_encoder_bert]

  # tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
  tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

  relevent_contexts = []

  if training_args.do_train:
    for question, context in tqdm(zip(questions, contexts)):
      relevent_context = search_with_elastic(es, question, data_args,training_args, q_encoder, tokenizer, context)
      relevent_contexts.append(relevent_context)
  else:
    for question in tqdm(questions):
      relevent_context = search_with_elastic(es, question, data_args,training_args, q_encoder, tokenizer)
      relevent_contexts.append(relevent_context)

  if training_args.do_eval:
    df = pd.DataFrame({'id':ids, 'question':questions, 'context':relevent_contexts, 'answers':answers})########EVAL########
  elif training_args.do_predict:
    df = pd.DataFrame({'id':ids, 'question':questions, 'context':relevent_contexts}) ###PREDICTION###
  elif training_args.do_train:
    df = pd.DataFrame({'id':ids, 'question':questions, 'context':relevent_contexts, 'answers':answers})

  if training_args.do_train:
    datasets = DatasetDict({"train": Dataset.from_pandas(df), 'validation':[]})
  else:
    datasets = DatasetDict({"validation": Dataset.from_pandas(df)})
  breakpoint()

  return datasets



def search_with_elastic(
  es: Elasticsearch, 
  question: str, 
  data_args: DataTrainingArguments,
  training_args: TrainingArguments,
  q_encoder: Optional[AutoModel] = None,
  tokenizer: Optional[AutoTokenizer] = None,
  context: Optional[str] = None,
  )->str:

  
  if data_args.use_entity_enrichment:
    match_phrases = create_match_phrases(question)
    query={
      'query':{
        'bool':{
          'should':[
            {
              'match':{
                'text':question,
              }
            },
            *match_phrases
          ]
        }
      }
    }
  elif data_args.eval_retrieval == 'elastic_dense':
    with torch.no_grad():
      q_encoder[0].eval()
      q_encoder[1].eval()
      q_tokenized = tokenizer([question], padding="max_length", truncation=True, return_tensors='pt', max_length=510, return_token_type_ids=False).to('cuda')
      q_emb_sen = q_encoder[0](**q_tokenized)
      q_emb_bert = q_encoder[1](**q_tokenized)
      q_output_sen = q_emb_sen[0].cpu().detach().numpy().tolist()
      q_output_bert = q_emb_bert[0].cpu().detach().numpy().tolist()
    
    # ###ENITIY ENRICHMENT###
    # match_phrases = create_match_phrases(question)
    # ######
    # query = {
    #   'query':{
    #     "script_score": {
    #       "query" : {
    #         'bool' : {
    #           'should' : [
    #             {
    #               "match" : {
    #                 "text": question
    #               },
    #             },
    #             *match_phrases
    #           ]
    #         } 
    #       },
    #       "script": {
    #         # "source": "cosineSimilarity(params.queryVector, doc['vector'])",
    #         "source": "_score * cosineSimilarity(params.queryVector, doc['vector']) / (_score + cosineSimilarity(params.queryVector, doc['vector'])) + cosineSimilarity(params.queryVector, doc['vector']) * _score / (_score + cosineSimilarity(params.queryVector, doc['vector']))",
    #         "params": {
    #           "queryVector": q_output
    #         }
    #       }
    #     }
    #   }
    # }
    pre_query = es.search(
      index='wiki_documents_dense_sen', 
      body={
        'query':{
          'match':{
            'text':question,
            # 'tokenizer': 'nori'
          }
        }
      },
      size=1
    )

    pre_query_sen = es.search(
      index='wiki_documents_dense',
      body={
        'query':{
          "script_score": {
            "query" : {
              "match" : {
                "text": question
              }
            },
            "script": {
              "source": "cosineSimilarity(params.queryVector_sen, doc['vector_sen']) + 1.0",
              "params": {
                "queryVector_sen": q_output_sen,
              }
            }
          }
        }
      },
      size= 1
    )

    pre_query_bert = es.search(
      index='wiki_documents_dense',
      body={
        'query':{
          "script_score": {
            "query" : {
              "match" : {
                "text": question
              }
            },
            "script": {
              "source": "cosineSimilarity(params.queryVector_bert, doc['vector_bert']) + 1.0",
              "params": {
                "queryVector_bert": q_output_bert,
              }
            }
          }
        }
      },
      size= 1
    )

    max_bm_score=pre_query['hits']['hits'][0]['_score']
    max_sen_score=pre_query_sen['hits']['hits'][0]['_score']
    max_bert_score=pre_query_bert['hits']['hits'][0]['_score']
    
    query = {
      'query':{
        "script_score": {
          "query" : {
            "match" : {
              "text": question
            },
          },
          "script": {
            # "source": "cosineSimilarity(params.queryVector, doc['vector'])",
            # "source": "_score * cosineSimilarity(params.queryVector, doc['vector']) / (_score + cosineSimilarity(params.queryVector, doc['vector']))",
            # "source": "_score * cosineSimilarity(params.queryVector, doc['vector']) / (_score + cosineSimilarity(params.queryVector, doc['vector'])) + cosineSimilarity(params.queryVector, doc['vector']) * _score / (_score + cosineSimilarity(params.queryVector, doc['vector']))",
            # "source": "_score * cosineSimilarity(params.queryVector, doc['vector'])",

            # BERT, Roberta 같이 사용할시에 사용할 
            # "source": "(cosineSimilarity(params.queryVector_sen, doc['vector_sen'])+1.0) * (cosineSimilarity(params.queryVector_bert, doc['vector_bert'])+1.0)",
            # "source": "_score *(cosineSimilarity(params.queryVector_sen, doc['vector_sen']) + cosineSimilarity(params.queryVector_bert, doc['vector_bert']))",
            # "source": "_score",
            # "source": "cosineSimilarity(params.queryVector_sen, doc['vector_sen']) + 1.0",
            # "source": "cosineSimilarity(params.queryVector_sen, doc['vector_bert']) + 1.0",

            # "source": "_score * cosineSimilarity(params.queryVector_sen, doc['vector_sen']) * cosineSimilarity(params.queryVector_bert, doc['vector_bert'])",
            # "source": "_score/params.max_bm_score + cosineSimilarity(params.queryVector_sen, doc['vector_sen']) + cosineSimilarity(params.queryVector_bert, doc['vector_bert'])",
            # "source": "_score/params.max_bm_score * cosineSimilarity(params.queryVector_sen, doc['vector_sen']) * cosineSimilarity(params.queryVector_bert, doc['vector_bert'])",
            
            "source": "_score/params.max_bm_score + (cosineSimilarity(params.queryVector_sen, doc['vector_sen'])+1.0)/params.max_sen_score + (cosineSimilarity(params.queryVector_bert, doc['vector_bert'])+1.0)/params.max_bert_score",
            "params": {
              "queryVector_sen": q_output_sen,
              "queryVector_bert": q_output_bert,
              "max_bm_score": max_bm_score,
              "max_sen_score": max_sen_score,
              "max_bert_score": max_bert_score,
            }
          }
        }
      }
    }
  else:
    query = {
      'query': {
        'match': {
          'text': question
        }
      }
    }

  if data_args.eval_retrieval == 'elastic_dense':
    res = es.search(index='wiki_documents_dense', body=query, size=data_args.top_k_retrieval)
  else:
    res = es.search(index='wiki_documents', body=query, size=data_args.top_k_retrieval)
  
  relevent_contexts = ''
  if training_args.do_train:
    relevent_contexts = context
  max_score = res['hits']['hits'][0]['_score']
  
  for i in range(data_args.top_k_retrieval):
    score = res['hits']['hits'][i]['_score']
    if score > max_score * 0.90:#######################################
      relevent_contexts += res['hits']['hits'][i]['_source']['text']
      relevent_contexts += ' '
    else:
      break
  
  return relevent_contexts

def create_match_phrases(question)-> List:
  ner = pororo.Pororo(task='ner', lang='ko')
  nered_question = ner(question)

  match_phrases = []

  for word in nered_question:
    if word[1] != 'O':
      match_phrases.append({'match_phrase':{'text': word[0]}})
  
  return match_phrases



