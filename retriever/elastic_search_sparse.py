import os
import sys
import re
import elasticsearch
import json
from elasticsearch import Elasticsearch, helpers
import pandas as pd
from datasets import DatasetDict, Dataset
from pandas.core.frame import DataFrame
import torch
from tqdm import tqdm
from transformers import TrainingArguments, AutoModel, AutoTokenizer

import time
from typing import List, Tuple, Optional

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)
from retriever import model_encoder

if os.path.dirname(os.path.abspath(os.path.dirname(__file__))) in sys.path :
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from preprocessor import Preprocessor

def create_elastic_object() -> Elasticsearch:
    """Elasticsearch와 연결하는 object 생성
    Returns:
        Elasticsearch: Elasticsearch Object
    """
    es = Elasticsearch([{"host": "localhost", "port": 9200}])

    if es.ping():
        print("Elasticsearch object created.")
    else:
        print("Failed to make Elasticsearch object")
    return es


def generator(contexts, index_name):
    for idx, context in enumerate(contexts):
        yield {
            "_index": index_name,
            "_type": "_doc",
            "_id": idx,
            "_source": {"text": context},
        }
    raise StopIteration

def generator_dense(text, title, document_id, p_embs_sen, p_embs_bert):
    for text_el, title_el, document_id_el, p_emb_sen, p_emb_bert in zip(text,title,document_id, p_embs_sen, p_embs_bert):
        yield {
        '_index': 'wiki_documents_dense',
        '_type': '_doc',
        '_id': document_id_el,
        '_source': {
            'text': text_el,
            'vector_sen': p_emb_sen,
            'vector_bert': p_emb_bert
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

    models_dir = filtpath  #'./retriever/ElasticSearchConfig'
    files = os.listdir(models_dir)
    files = sorted(files)

    for i, d in enumerate(files, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory you want to load: ")

    config_file = os.path.abspath(os.path.join(models_dir, files[int(d_idx)]))
    print("Elastic search config file is: {}".format(config_file))

    with open(config_file) as f:
        lines = "".join(f.readlines())
    ESconfig = re.sub("\\|\n|  ", "", lines)
    return ESconfig

def dense_embedding(df: DataFrame)-> List:
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    p_encoder_sen = model_encoder.RobertaEncoder.from_pretrained('encoders/p_encoder_neg_sen').to('cuda')
    p_encoder_bert = model_encoder.BertEncoder.from_pretrained('encoders/p_encoder').to('cuda')
    p_embs_sen = []
    p_embs_bert = []
    for index, document in tqdm(df.iterrows()):
        with torch.no_grad():
            p_encoder_sen.eval()
            p_encoder_bert.eval()
            p_val_sen = tokenizer([document['text']], padding="max_length", truncation=True, return_tensors='pt', max_length=510, return_token_type_ids=False).to('cuda')
            p_val_bert = tokenizer([document['text']], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            p_emb_sen = p_encoder_sen(**p_val_sen)
            p_emb_bert = p_encoder_bert(**p_val_bert)
            
            p_embs_sen.append(p_emb_sen[0].cpu().detach().numpy().tolist())
            p_embs_bert.append(p_emb_bert[0].cpu().detach().numpy().tolist())
    return p_embs_sen, p_embs_bert

def prepare_sparse_config(es, docs_config, index_name, data_args):
    es_indices = es.indices.get_alias("*").keys()
    if index_name in es_indices:
        print(f"{index_name} already exists.")
        es.indices.delete(index=index_name, ignore=[400, 404])

    if data_args.reconfig:
        es.indices.create(index=index_name, body=docs_config, ignore=400)

        with open(os.path.join("/opt/ml/data/wikipedia_documents.json") , "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
            )  # set 은 매번 순서가 바뀌므로
        
        if data_args.preprocessing_pattern != None:
            contexts = Preprocessor.preprocessing(contexts, pt_num=data_args.preprocessing_pattern)

        gen = generator(contexts[: 100 * (len(contexts) // 100)], index_name)
        try:
            helpers.bulk(es, gen, chunk_size=100)
        except Exception as e:
            print("Done")
        gen = generator(contexts[100 * (len(contexts) // 100) :], index_name)
        try:
            helpers.bulk(es, gen, chunk_size=1)
        except Exception as e:
            print("Done")

    return es

def run_elastic_sparse_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
) -> DatasetDict:

    index_name = "wiki_documents"

    es = create_elastic_object()
    if data_args.reconfig:
        config = eval(select_ESconfig("./retriever/ElasticSearchConfig"))
        print("Start create index")
        es = prepare_sparse_config(es, config, index_name, data_args)
        time.sleep(10)

    print(es.indices.analyze(
            index=index_name, body={"analyzer": "nori_analyzer", "text": "동해물과 백두산이"})
    )
    
    total = []
    exact_count = 0
    for example in tqdm(datasets["validation"]):
        relevent_context = search_with_elastic(
            es, example["question"], index_name, data_args
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
    q_encoder_sen: Optional[AutoModel] = None,
    q_encoder_bert: Optional[AutoModel] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> str:
    query = {"query": {"match": {"text": question}}}
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
