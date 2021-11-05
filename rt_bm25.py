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

import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from preprocessor import Preprocessor

from datasets import (
    Dataset,
    Value,
    Sequence,
    Features,
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
        data_path: Optional[str] = '/opt/ml/data',
        context_path: Optional[str] = "wikipedia_documents.json",
        pt_num: Optional[str] = None,
        add_special_tokens_flag : Optional[bool] = False
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
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
            )  # set 은 매번 순서가 바뀌므로

        self.add_special_tokens_flag = add_special_tokens_flag
        if self.pt_num != None:
            # self.contexts = list(map(lambda x : Preprocessor.preprocessing(data = x, pt_num=self.pt_num),self.contexts)) # Preprocessor.preprocessing(data = x, pt_num=self.pt_num)
            self.contexts = Preprocessor.preprocessing(self.contexts, pt_num=self.pt_num)
        
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # corpus wiki 데이터를 전처리 합니다.
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다.
        self.indexer = None  # build_faiss()로 생성합니다.

        ## BM25 추가용 ##
        self.BM25 = None
        self.tokenizer = tokenize_fn

    def get_sparse_BM25(self) -> NoReturn:

        """
            Passage Embedding을 만들고 TFIDF와 Embedding을 pickle로 저장
            만약 미리 저장된 파일이 있으면 저장된 pickle을 호출
        """

        # Pickle을 저장 "0123"
        pt_num_sorted = "".join(sorted(self.pt_num)) if self.pt_num else "raw"
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
            tokenized_contexts = [self.tokenizer(i) for i in tqdm(self.contexts)]
            self.BM25 = BM25Plus(tokenized_contexts)
            with open(bm_emd_path, "wb") as file:
                pickle.dump(self.BM25, file)
            print("BM25_class_instant pickle saved.")

    def retrieve_train_BM25(
        self, dataset: Union[str, Dataset], topk: Optional[int] = 1, rtt_name : Optional[str] = None
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """ 
            Dataset의 Question을 BM25를 이용해서 유사한 Wikipedia Data를 불러와서 기존 Train Data의 context와 합치고 answer start point를 재조정
        """
        assert self.BM25 is not None and isinstance(dataset, Dataset)

        sep_flag = 1 if self.add_special_tokens_flag == True else 0 # 서로 다른 Passage를 구분하는 special token을 넣을 지 결정하는 flag
        rtt_flag = 1 if rtt_name != None else 0 # rtt data를 사용할 지 결정하는 flag
        json_name = f"train_retrieval_{self.pt_num}_{sep_flag}_{rtt_flag}_topk{topk}.json"
        json_path = os.path.join('./json', json_name)

        if os.path.isfile(json_path): # json file이 이미 존재하면 이를 불러와서 Dataframe를 만들고 반환합니다.
            print("Load Saved Retrieval Json Data.")
            with open(json_path , "r", encoding="utf-8") as f:
                json_data = json.load(f)
            cqas = pd.DataFrame(json_data) 
        else : # json file이 존재 하지 않는다면 BM25를 이용해서 train dataset을 재구성
            total = []
            print('Make Retrieval Pandas Data')
            with timer("query exhaustive search"):
                doc_scores, doc_indices, doc_rank = self.get_relevant_train_bulk_BM25(dataset, k=topk, )

            for idx, example in enumerate(
                tqdm(dataset, desc="BM25 retrieval: ")
            ):

                gap_size = 9 if self.add_special_tokens_flag == True else 1 # 문단 사이사이에 ' [SPLIT] ' 이 들어가기 때문에 길이 9가 추가되어야 합니다.
 
                # 원래 문단 앞에 passage가 추가 되어야 하면 answer start point를 재조정합니다.
                doc_start = 0 
                if doc_rank[idx] > 0 :
                    for i in range(doc_rank[idx]) :
                        doc_id = doc_indices[idx][i]
                        doc_context = self.contexts[doc_id]
                        doc_start += (len(doc_context) + gap_size)

                    answer = example['answers']
                    answer_start, answer_text = answer['answer_start'][0], answer['text'][0]
                    example['answers'] = {'answer_start' : [doc_start + answer_start], 'text' : [answer_text]}       

                split_string = " [SPLIT] " if self.add_special_tokens_flag else " "

                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": split_string.join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            cqas.to_json(json_path)

        f = Features(
            {
                "answers": Sequence(
                feature={
                    "text": Value(dtype="string", id=None),
                    "answer_start": Value(dtype="int32", id=None),
                },
                length=-1,
                id=None,
            ),
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
            }
        )
        print('Make Retrieved Train Dataset')
        datasets = Dataset.from_pandas(cqas, features=f)
        return datasets
        
    def get_relevant_train_bulk_BM25(
        self, datasets: Dataset, k: Optional[int] = 1, 
    ) -> Tuple[List, List]:
        """
            BM25를 활용해서 datasets의 question을 이용해 Wikipedia passage들을 가져오고 context와 합치는 함수
        """
        print("Build BM25 score, indices")

        data_size = len(datasets)
        queries = datasets['question']
        contexts = datasets['context']

        tokenized_queries= [self.tokenizer(i) for i in queries]        
        doc_scores = []
        doc_indices = []
        doc_ranks = []

        for i in tqdm(range(data_size)):
            scores = self.BM25.get_scores(tokenized_queries[i])
            context_txt = contexts[i]
            sorted_score = np.sort(scores)[::-1]
            sorted_id = np.argsort(scores)[::-1]
            
            selected_scores = []
            selected_indices = []

            org_id = self.contexts.index(context_txt)

            j = 0
            while(j < k) :
                selected_scores.append(sorted_score[j])
                selected_indices.append(sorted_id[j])
                j += 1

            if org_id not in selected_indices : # top k 안에 train context가 포함 되지 않는다면 마지막에 context를 넣어줍니다.
                doc_ranks.append(j)
                selected_scores.append(0)
                selected_indices.append(org_id)
            else : # top k 안에 train context가 포함되면 몇 번째인지를 파악합니다.
                org_rank = selected_indices.index(org_id)
                doc_ranks.append(org_rank)

            doc_scores.append(selected_scores)
            doc_indices.append(selected_indices)
        return doc_scores, doc_indices, doc_ranks
    

    def retrieve_BM25(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: Optional[int] = 1,
        score_ratio: Optional[float] = None,
        pickle_path: Optional[str] = ''
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
        pickle_name = f"BM25_retrieve_{pickle_path}.bin"
        if os.path.isfile(pickle_name) :
            with open(pickle_name, "rb") as file:
                cqas = pickle.load(file)
            print("BM25 retrieve pickle load.")
            return cqas
        else :
            if isinstance(query_or_dataset, str):
                doc_scores, doc_indices = self.get_relevant_doc_BM25(
                    query_or_dataset, k=topk
                )
                print("[Search query]\n", query_or_dataset, "\n")

                for i in range(topk):
                    print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                    print(self.contexts[doc_indices[i]])

                return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

            elif isinstance(query_or_dataset, Dataset):

                # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
                total = []
                with timer("query exhaustive search"):
                    doc_scores, doc_indices = self.get_relevant_doc_bulk_BM25(query_or_dataset, k=topk, score_ratio=score_ratio)
                for idx, example in enumerate(
                    tqdm(query_or_dataset, desc="BM25 retrieval: ")
                ):
                    split_string = " [SPLIT] " if self.add_special_tokens_flag else " "

                    tmp = {
                        # Query와 해당 id를 반환합니다.
                        "question": example["question"],
                        "id": example["id"],
                        # Retrieve한 Passage의 id, context를 반환합니다.
                        "context_id": doc_indices[idx],
                        "context": split_string.join([self.contexts[pid] for pid in doc_indices[idx]])
                    }
                    if "context" in example.keys() and "answers" in example.keys():
                        # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    total.append(tmp)

                cqas = pd.DataFrame(total)
                if not pickle_path :
                    with open(pickle_name, "wb" ) as file:
                        pickle.dump(cqas, file)
                return cqas
            
    def get_relevant_doc_BM25(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

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

        # score 높은순으로 index 정렬합니다.
        doc_indices = np.argsort(-doc_scores)
        return doc_scores[doc_indices[:k]], doc_indices[:k]

    def get_relevant_doc_bulk_BM25(
        self, query_or_dataset: Union[str, Dataset], k: Optional[int] = 1, score_ratio: Optional[float] = 0
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
        query = query_or_dataset['question']
        tokenized_queries= [self.tokenizer(i) for i in query] 
        doc_scores = []
        doc_indices = []
        for i in tqdm(tokenized_queries):
            scores = self.BM25.get_scores(i)
            sorted_score = np.sort(scores)[::-1]
            sorted_id = np.argsort(scores)[::-1]
            boundary = []

            ## 해당 query의 가장 높은 score(sorted_score[0])의 x score_ratio까지의 context만 바.
            for z in sorted_score:
                if z >= sorted_score[0] * score_ratio:
                    boundary.append(True)
                else:
                    boundary.append(False)

            if len(sorted_score[boundary]) <= k:
                doc_scores.append(sorted_score[boundary])
                doc_indices.append(sorted_id[boundary])
            else:
                doc_scores.append(sorted_score[:k])
                doc_indices.append(sorted_id[:k])
        
        # Validation시 recall@K를 출력하게 합니다.
        if 'answers' in  query_or_dataset.column_names :
            print(f'** Calculating Recall@{k}')
            cnt = 0
            for i, q in enumerate(query_or_dataset['context']) :
                for wiki_idx in list(doc_indices[i]) :
                    if q == self.contexts[wiki_idx]:
                        cnt += 1
                        break
            total_len = len(query_or_dataset['context'])
            print(f'** Recall@{k} = {cnt / total_len: .4f}, Count:{cnt}')
        return doc_scores, doc_indices
