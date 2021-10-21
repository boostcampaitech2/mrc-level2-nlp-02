import elasticsearch
from elasticsearch import Elasticsearch
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments
import pororo
from tqdm import tqdm

from typing import List

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)


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

  relevent_contexts = []
  if data_args.use_entity_enrichment:
    print('Use entitiy enrichment.')

  for question in tqdm(questions):
    relevent_context = search_with_elastic(es, question, data_args)
    relevent_contexts.append(relevent_context)

  df = pd.DataFrame({'id':ids, 'question':questions, 'context':relevent_contexts})
  datasets = DatasetDict({"validation": Dataset.from_pandas(df)})

  return datasets

def search_with_elastic(es: Elasticsearch, question: str, data_args: DataTrainingArguments)->str:

  
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
  else:
    query = {
      'query': {
        'match': {
          'text': question
        }
      }
    }

  res = es.search(index='wiki_documents', body=query, size=data_args.top_k_retrieval)
  
  relevent_contexts = ''
  max_score = res['hits']['hits'][0]['_score']
  
  for i in range(data_args.top_k_retrieval):
    score = res['hits']['hits'][i]['_score']
    if score > max_score * 0.85:
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




