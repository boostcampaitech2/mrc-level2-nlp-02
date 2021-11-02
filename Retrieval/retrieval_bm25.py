import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Plus
from konlpy.tag import Mecab

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from sklearn.feature_extraction.text import TfidfVectorizer
import re

from preprocessor import Preprocessor
from transformers import BertTokenizerFast

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        contexts : list ,
    ) -> NoReturn:

        self.contexts = []
        self.ids = []

        for i in range(len(contexts)) :
            self.ids.append(i)
            self.contexts.append(contexts[i])
        print("Context Length : %d " %len(self.contexts))

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.BM25 = None
        self.tokenizer = tokenize_fn
        
    def get_sparse_BM25(self, dir_path, name) -> NoReturn:
        pickle_name = f"BM25_embedding_{name}.bin"
        bm_emd_path = os.path.join(dir_path, pickle_name)

        if os.path.isfile(bm_emd_path):
            with open(bm_emd_path, "rb") as file:
                self.BM25 = pickle.load(file)    
            print("BM25 Embedding pickle %s loaded." %bm_emd_path)

        else:
            print("Build passage BM25_class_instant")
            tokenized_contexts= [self.tokenizer(i) for i in tqdm(self.contexts)]
            self.BM25 = BM25Plus(tokenized_contexts)           
            with open(bm_emd_path, "wb") as file:
                pickle.dump(self.BM25, file)
            print("BM25_class_instant pickle saved.")

    def retrieve_BM25(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 10
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.BM25 is not None and isinstance(query_or_dataset, Dataset)
        
        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        with timer("query exhaustive search"):
            doc_scores, doc_indices = self.get_relevant_doc_BM25(query_or_dataset['question'], k=topk)
        for idx, example in enumerate(
            tqdm(query_or_dataset, desc="BM25 retrieval: ")
        ):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": doc_indices[idx],
                "context" : example["context"]
            }
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

    def get_relevant_doc_BM25(
        self, queries: List, k: Optional[int] = 10, score_ratio: Optional[float] = None
    ) -> Tuple[List, List]:

        print("Build BM25 score, indices")
        tokenized_queries= [self.tokenizer(i) for i in queries]        
        doc_scores = []
        doc_indices = []
        for i in tqdm(tokenized_queries):
            scores = self.BM25.get_scores(i)
            sorted_score = np.sort(scores)[::-1]
            sorted_id = np.argsort(scores)[::-1]

            doc_scores.append(sorted_score[:k])
            doc_indices.append(sorted_id[:k])
        return doc_scores, doc_indices