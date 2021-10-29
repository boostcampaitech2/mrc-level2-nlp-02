import os
import sys
import re
import elasticsearch
from elasticsearch import Elasticsearch, helpers
import pandas as pd
from datasets import DatasetDict, Dataset
import torch
from transformers import TrainingArguments, AutoModel, AutoTokenizer
from typing import List, Tuple, Optional

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)
if os.path.dirname(os.path.abspath(os.path.dirname(__file__))) in sys.path :
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from retriever.model_encoder import BertEncoder, RobertaEncoder

import time
from contextlib import contextmanager
from tqdm.auto import tqdm

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[**{name}] done in {time.time() - t0:.3f} s")

def create_elastic_object() -> Elasticsearch:
    """Elasticsearch와 연결하는 object 생성
    Returns:
        Elasticsearch: Elasticsearch Object
    """
    es = Elasticsearch([{"host": "localhost", "port": 9200}])

    if es.ping():
        print("**Elasticsearch object created.")
    else:
        print("**Failed to make Elasticsearch object")
    return es


def generator(text, document_id, p_embs_not_splited, index_name):
    for text_el, document_id_el, p_embs_el in zip(text,document_id, p_embs_not_splited):
        yield {
            '_index': index_name,
            '_type': '_doc',
            '_id': document_id_el,
            '_source': {
                'text': text_el,
                'vector': p_embs_el
            }
        }
    raise StopIteration


def select_ESconfig(filtpath: str) -> str:
    """[summary]
    ES config file 선택하여 적용하기
    Args:
        filtpath (str):
            file path

    Returns:
        str: [description]
            wiki에 적용할 config
    """
    config_file = os.path.join(filtpath, 'DenseEnsemble.txt')
    print("**Elastic search config file is: {}".format(config_file))

    with open(config_file) as f:
        lines = "".join(f.readlines())
    ESconfig = re.sub("\\|\n|  ", "", lines)
    return ESconfig


def prepare_ensemble_config(es, docs_config, index_name, tokenizer, p_dir, args):

    es_indices = es.indices.get_alias("*").keys()
    if index_name in es_indices:
        print(f"**{index_name} already exists.")
        es.indices.delete(index=index_name, ignore=[400, 404])

    if args.reconfig:
        es.indices.create(index=index_name, body=docs_config, ignore=400)

        df = pd.read_json("/opt/ml/data/wikipedia_documents.json").T
        df = df.drop_duplicates(subset=["text"])
        text = df['text'].to_list()
        document_id = df['document_id'].to_list()

        with timer("wiki dense embedding"):
            p_encoder = RobertaEncoder.from_pretrained(p_dir).to('cuda')
            p_embs_not_splited = []
            for index, document in tqdm(df.iterrows()):
                with torch.no_grad():
                    p_encoder.eval()
                    p_val = tokenizer([document['text']], padding="max_length", truncation=True, return_tensors='pt', max_length=510, return_token_type_ids=False).to('cuda')
                    p_emb = p_encoder(**p_val)
                    p_embs_not_splited.append(p_emb[0].cpu().detach().numpy().tolist())

        with timer("index reconfig"):
            L100idx= 100*(len(df) // 100)
            gen = generator(text[:L100idx], document_id[:L100idx], p_embs_not_splited[:L100idx], index_name)
            try:
                helpers.bulk(es, gen, chunk_size=100)
            except Exception as e:
                pass
            gen = generator(text[L100idx:], document_id[L100idx:], p_embs_not_splited[L100idx:], index_name)
            try:
                helpers.bulk(es, gen, chunk_size=1)
            except Exception as e:
                pass
    return es

def run_elastic_ensemble_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
) -> DatasetDict:
    
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    encoders = os.listdir(data_args.dense_rt_path)
    assert len(encoders) == 2 ## 같은 dense encoder는 하나의 dir에 p_encoder, q_encoder 하나씩만 넣어주세요
    
    index_name = "wiki_documents_ensemble"
    es = create_elastic_object()
    
    p_dir = os.path.join(data_args.dense_rt_path, encoders[0] if 'p' in encoders[0] else encoders[1])
    q_dir = os.path.join(data_args.dense_rt_path, encoders[0] if 'q' in encoders[0] else encoders[1]) 
    if data_args.reconfig:
        config = eval(select_ESconfig("./retriever/ElasticSearchConfig"))    
        print("**Start create index")
        es = prepare_ensemble_config(es, config, index_name, tokenizer, p_dir, data_args)
    
    print('**Load query encoder')
    q_encoder = RobertaEncoder.from_pretrained(q_dir).to('cuda')
    total = []
    exact_count = 0
    for example in tqdm(datasets["validation"]):
        relevent_context = search_with_elastic(
            es, example["question"], index_name, data_args, q_encoder, tokenizer
        )
        tmp = {
            "question": example["question"],
            "id": example["id"],
            "context": relevent_context,
        }
        if "context" in example.keys() and "answers" in example.keys():
            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
            tmp["original_context"] = example["context"]
            tmp["answers"] = example["answers"]
            if example["context"] in relevent_context:
                exact_count += 1
        total.append(tmp)

    Total_count = len(datasets["validation"])
    print(f"**** Accuracy: {exact_count / Total_count * 100:.3f}")
    print(f"**** Correct count: {exact_count}")
    df = pd.DataFrame(total)
    datasets = DatasetDict({"validation": Dataset.from_pandas(df)})
    return datasets


def search_with_elastic(
    es: Elasticsearch,
    question: str,
    index_name: str,
    data_args: DataTrainingArguments,
    q_encoder: Optional[AutoModel] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> str:
    
    with torch.no_grad():
        q_encoder.eval()
        q_val = tokenizer(question, padding="max_length", truncation=True, return_tensors='pt', max_length = 510, return_token_type_ids=False).to('cuda')
        q_emb = q_encoder(**q_val)
        q_output = q_emb[0].cpu().detach().numpy().tolist()

    query = {
        'query':{
            "script_score": {
                "query" : {
                    "match" : {
                        "text": question
                    }
                },
                "script": {
                    # "source": "1 / (1 + l2norm(params.queryVector, doc['vector']))",
                    # "source": "cosineSimilarity(params.queryVector, doc['vector']) + 1.0",
                    # "source": "_score + cosineSimilarity(params.queryVector, doc['vector'])",
                    "source": "_score * cosineSimilarity(params.queryVector, doc['vector']) / (_score + cosineSimilarity(params.queryVector, doc['vector']))",
                    "params": {
                        "queryVector": q_output
                        }
                    }
                }
            }
        }
    res = es.search(index=index_name, body=query, size=data_args.top_k_retrieval)

    relevent_contexts = ""
    max_score = res["hits"]["hits"][0]["_score"]

    for i in range(data_args.top_k_retrieval):
        score = res["hits"]["hits"][i]["_score"]
        if score > max_score * data_args.score_ratio:
            relevent_contexts += res["hits"]["hits"][i]["_source"]["text"]
            relevent_contexts += " "
        else:
            break
    return relevent_contexts
