import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Plus

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Optional, Union

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessor import Preprocessor

from datasets import (
    Dataset,
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
        data_path: Optional[str] = '../data',
        context_path: Optional[str] = "wikipedia_documents.json",
        pt_num: Optional[str] = None,
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        self.pt_num = pt_num
        with open(os.path.join(data_path,context_path) , "r", encoding="utf-8") as f:
            wiki = json.load(f)

        if self.pt_num != None:
            self.contexts = list(map(lambda x : Preprocessor.preprocessing(data = x, pt_num=self.pt_num),self.contexts))
        
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
            )  # set 은 매번 순서가 바뀌므로

        print(f"Lengths of unique contexts : {len(self.contexts)}")

        #corpus wiki 데이터를 전처리 합니다.
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.
        
        ## BM25 추가용 ##
        self.BM25 = None
        self.tokenizer = tokenize_fn

    def get_sparse_BM25(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장 "0123"
        pt_num_sorted = "".join(sorted(self.pt_num)) if self.pt_num else 'raw'
        pickle_name = f"BM25_embedding_{pt_num_sorted}.bin"
        bm_emd_path = os.path.join(self.data_path, pickle_name)

        # BM25 존재하면 가져오기
        if os.path.isfile(bm_emd_path):
            with open(bm_emd_path, "rb") as file:
                self.BM25 = pickle.load(file)            
            print("BM25 Embedding pickle load.")
        
        # https://github.com/dorianbrown/rank_bm25 -> initalizing 부분
        # BM25 존재 하지 않으면, tokenizer 한 후, BM25Plus로 passage embedding
        else:
            print("Build passage BM25_class_instant")
            # BM25는 어떤 text 전처리 X ->  BM25 클래스의 인스턴스를 생성
            tokenized_contexts= [self.tokenizer(i) for i in tqdm(self.contexts)]
            self.BM25 = BM25Plus(tokenized_contexts)           
            with open(bm_emd_path, "wb") as file:
                pickle.dump(self.BM25, file)
            print("BM25_class_instant pickle saved.")

    def retrieve_BM25(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, score_ratio: Optional[float] = None
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.BM25 is not None, "get_sparse_BM25() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_BM25(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_BM25(query_or_dataset['question'], k=topk, score_ratio=score_ratio)
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="BM25 retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_BM25(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        tokenized_query = self.tokenizer(query) 
        
        # ex. array([2.77258872, 5.3162481 , 2.77258872])
        # 자동으로 passage embedding과 query vector간의 계산 완료!
        doc_scores = self.BM25.get_scores(tokenized_query)
        
        # score 높은순으로 index 정렬
        doc_indices=np.argsort(-doc_scores) 
        return doc_scores[doc_indices[:k]], doc_indices[:k]

    def get_relevant_doc_bulk_BM25(
        self, queries: List, k: Optional[int] = 1, score_ratio: Optional[float] = None
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        print("Build BM25 score, indices")
        tokenized_queries= [self.tokenizer(i) for i in queries]        
        doc_scores = []
        doc_indices = []
        for i in tqdm(tokenized_queries):
            scores = self.BM25.get_scores(i)
            boundary=[]
            sorted_score = np.sort(scores)[::-1]
            sorted_id = np.argsort(scores)[::-1]
            boundary = []  
            
            ## 해당 query의 가장 높은 score(sorted_score[0])의 x score_ratio까지의 점수만 받는다.
            if score_ratio is not None:
                for z in sorted_score:
                    if z>=sorted_score[0]*score_ratio:
                        boundary.append(True)
                    else:
                        boundary.append(False)        

                if len(sorted_score[boundary])<=k:
                    doc_scores.append(sorted_score[boundary])
                    doc_indices.append(sorted_id[boundary])
                else:
                    doc_scores.append(sorted_score[:k])
                    doc_indices.append(sorted_id[:k])
            else:
                doc_scores.append(sorted_score[:k])
                doc_indices.append(sorted_id[:k])
        return doc_scores, doc_indices
