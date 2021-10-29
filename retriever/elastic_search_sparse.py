import os
import sys
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
    es = Elasticsearch([{"host": "localhost", "port": 9200}])

    if es.ping():
        print("Elasticsearch object created.")
    else:
        print("Failed to make Elasticsearch object")
    return es


def generator(df, index_name):
    for c, line in enumerate(df):
        yield {
            "_index": index_name,
            "_type": "_doc",
            "_id": line.get("document_id", None),
            "_source": {"title": line.get("title", ""), "text": line.get("text", "")},
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


def prepare_sparse_config(es, docs_config, index_name, args):

    es_indices = es.indices.get_alias("*").keys()
    if index_name in es_indices:
        print(f"{index_name} already exists.")
        es.indices.delete(index=index_name, ignore=[400, 404])

    if args.reconfig:
        es.indices.create(index=index_name, body=docs_config, ignore=400)

        df = pd.read_json("/opt/ml/data/wikipedia_documents.json").T
        df = df.drop_duplicates(subset=["text"])
        df = df.to_dict("records")

        gen = generator(df[: 100 * (len(df) // 100)], index_name)
        try:
            helpers.bulk(es, gen, chunk_size=100)
        except Exception as e:
            print("Done")
        gen = generator(df[100 * (len(df) // 100) :], index_name)
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
    q_encoder: Optional[AutoModel] = None,
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
