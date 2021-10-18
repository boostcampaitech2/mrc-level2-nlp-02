import elasticsearch
from elasticsearch import Elasticsearch
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments

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
  for question in questions:
    relevent_context = search_with_elastic(es, question, data_args.top_k_retrieval)
    relevent_contexts.append(relevent_context)

  df = pd.DataFrame({'id':ids, 'question':questions, 'context':relevent_contexts})
  datasets = DatasetDict({"validation": Dataset.from_pandas(df)})

  return datasets

def search_with_elastic(es: Elasticsearch, question: str, size: int)->str:
  query = {
    'query': {
      'match': {
        'text': question
      }
    }
  }
  res = es.search(index='wiki_documents', body=query, size=size)
  
  relevent_contexts = ''
  for i in range(size):
    relevent_contexts += res['hits']['hits'][i]['_source']['text']
    relevent_contexts += ' '
  
  return relevent_contexts
  





