import os
import re
import elasticsearch
from elasticsearch import Elasticsearch, helpers
import pandas as pd
from datasets import DatasetDict, Dataset
import torch
from tqdm import tqdm
from transformers import TrainingArguments, AutoModel, AutoTokenizer

from typing import List, Tuple, Optional

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

def generator(df):
  for c, line in enumerate(df):
    yield {
      '_index': 'wiki_documents',
      '_type': '_doc',
      '_id': line.get('document_id', None),
      '_source': {
        'title': line.get('title', ''),
        'text': line.get('text', '')
      }
    }
  raise StopIteration

def select_ESconfig(filtpath :str
    ) -> Tuple[str, str]:
    """[summary]
    ES config file 선택하여 적용하기
    Args:
        filtpath (str): 
            file path

    Returns:
        Tuple[str, str]:
            Tuple[0]: wiki에 적용할 config
            Tuple[1]: query에 적용할 config          
    """
    models_dir = filtpath #'./retriever/ElasticSearchConfig'
    files = os.listdir(models_dir)
    files = sorted(files)

    for i, d in enumerate(files, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory you want to load: ")

    config_file = os.path.abspath(os.path.join(models_dir, files[int(d_idx)]))
    print("checkpoint_dir is: {}".format(config_file))
    
    with open(config_file) as f :
        lines = ''.join(f.readlines())
    evalform = re.sub('\\|\n|  ','', lines)
    ESconfig = evalform.split('##')
    return ESconfig

def prepare_config(es, docs_config, index_name, args) :
  reconfig_flag = False
  es_indices = es.indices.get_alias("*").keys()
  if index_name in es_indices :
    print(f"{index_name} already exists.")
    reconfig_flag = input("Do you want to recreate the index? (T or F): ")
    reconfig_flag = True if reconfig_flag == 'T' else False
  
  if index_name not in es_indices or reconfig_flag :
    print(es.indices.create(index=index_name, body=docs_config, ignore=400))

    df = pd.read_json('/opt/ml/data/wikipedia_documents.json').T
    df = df.drop_duplicates(subset=['text'])
    df = df.to_dict('records')
    gen = generator(df)

    try:
      helpers.bulk(es, gen, chunk_size = 1)
    except Exception as e:
      print('Done')
  return es

def run_elastic_sparse_retrieval(
  datasets: DatasetDict,
  training_args: TrainingArguments,
  data_args: DataTrainingArguments,
  ) -> DatasetDict:

  index_name = 'wiki_documents'
  docs_config, query_config = select_ESconfig('./retriever/ElasticSearchConfig')

  es = create_elastic_object()
  print(docs_config)
  print(query_config)
  breakpoint()
  
  docs_config = eval(docs_config)
  # es.indices.put_settings(body=docs_config['settings'], index='wiki_documents')
  # es.indices.put_settings(body=['mappings'], index='wiki_documents')
  
  if data_args.reconfig :
    print("Start create index")  
    es = prepare_config(es, docs_config, index_name, data_args)
    
  questions = datasets['validation']['question']
  ids = datasets['validation']['id']
  relevent_contexts = []

  for question in tqdm(questions):
    query = eval(query_config)
    relevent_context = search_with_elastic(es, query, index_name, data_args)
    relevent_contexts.append(relevent_context)

  df = pd.DataFrame({'id':ids, 'question':questions, 'context':relevent_contexts})
  datasets = DatasetDict({"validation": Dataset.from_pandas(df)})

  return datasets

def search_with_elastic(
  es: Elasticsearch, 
  query: dict,
  index_name: str, 
  data_args: DataTrainingArguments,
  q_encoder: Optional[AutoModel] = None,
  tokenizer: Optional[AutoTokenizer] = None,
  )->str:

  breakpoint()
  res = es.search(index=index_name, body=query, size=data_args.top_k_retrieval)
  

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