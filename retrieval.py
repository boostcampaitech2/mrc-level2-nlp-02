from optparse import Option
import os
import json
from plistlib import Data
import time
import faiss
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)

from model_encoder import RobertaEncoder


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
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
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
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

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
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

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
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

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
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
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
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

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
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

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
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

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


class DenseRetrieval:
    def __init__(
        self,
        args,
        data_path: str,
        num_neg: int,
        tokenizer,
        p_encoder,
        q_encoder,
        save_dir: str,
    ):
        """[summary] : encoder를 활용해 query와 passage를 embedding하고 embedding된 vector를 유사도 검색에 활용
        Args:
            args : encoder를 train할 때 필요한 arguments
            data_path : 학습을 위한 MRC dataset의 path
            num_neg : Negative sampling 개수
            tokenizer : Pretrained tokenizer
            p_encoder : query를 embedding 할 때 사용할 모델
            q_encoder : passage를 embedding 할 때 사용할 모델
            save_dir : encoder 모델을 저장할 디렉토리 path
        """
        self.args = args

        self.train_dataset = load_from_disk(data_path)["train"]
        self.valid_dataset = load_from_disk(data_path)["validation"]
        self.p_dataset = self.train_dataset["context"] + self.valid_dataset["context"]
        self.q_dataset = self.train_dataset["question"] + self.valid_dataset["question"]

        self.num_neg = num_neg
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.train_dataloader = None
        self.valid_dataloader = None

        self.save_dir = save_dir

        if num_neg != 0:
            self.prepare_negative(num_neg=num_neg)
        else:
            self.prepare_in_batch()

    def prepare_in_batch(self, tokenizer=None):
        """
        [summary] : in_batch negative sampling 방식으로 학습한 dataset 구성
        """
        print("prepare in_batch dataset !!!")
        if tokenizer is None:
            tokenizer = self.tokenizer

        q_seqs = tokenizer(
            self.q_dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            # return_token_type_ids=False,
        )

        p_seqs = tokenizer(
            self.p_dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            # return_token_type_ids=False,
        )

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size,
        )

        valid_seqs = tokenizer(
            self.p_dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            # return_token_type_ids=False,
        )
        valid_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
            valid_seqs["token_type_ids"],
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.args.per_device_train_batch_size
        )

    def prepare_negative(self, num_neg=3, tokenizer=None):
        """
        [summary] : Negative Sampling을 통해 dataset을 구성
        """
        print("prepare negative sampling dataset !!!")
        if tokenizer is None:
            tokenizer = self.tokenizer

        corpus = np.array(list(set([example for example in self.p_dataset])))
        p_with_neg = []

        for p_positive in self.p_dataset:
            while True:
                neg_idx = np.random.randint(len(corpus), size=num_neg)

                if p_positive not in corpus[neg_idx]:
                    p_neg = corpus[neg_idx]

                    p_with_neg.append(p_positive)
                    p_with_neg.extend(p_neg)
                    break

        q_seqs = tokenizer(
            self.q_dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_neg + 1, max_len
        )

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size,
            drop_last=True,
        )

        valid_seqs = tokenizer(
            self.p_dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        valid_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.args.per_device_train_batch_size
        )

    def make_wiki_embedding(self, batch_size, args=None):
        if args is None:
            args = self.args

        df = pd.read_json("/opt/ml/data/wikipedia_documents.json").transpose()
        wiki_texts = list(df["text"])
        print(f"wiki text size : {len(wiki_texts)}")

        wiki_seqs = self.tokenizer(
            wiki_texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(args.device)

        wiki_dataset = TensorDataset(
            wiki_seqs["input_ids"],
            wiki_seqs["attention_mask"],
            wiki_seqs["token_type_ids"],
        )

        wiki_dataloader = DataLoader(wiki_dataset, batch_size=batch_size)

        self.p_encoder.to(args.device)

        with torch.no_grad():
            self.p_encoder.eval()

            p_embs = []

            for batch in tqdm(wiki_dataloader, unit="batch"):
                batch = tuple(t.to(args.device) for t in batch)
                wiki_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                p_emb = self.p_encoder(**wiki_inputs).to("cpu")
                p_embs.append(p_emb)

        p_embs = torch.cat(p_embs, dim=0).numpy()
        print(f"wiki passage embedding shape:{p_embs.shape}")
        print("\nMake wiki embedding csv file !!!")

        emb_df = pd.DataFrame(p_embs)

        emb_df.to_csv("/opt/ml/data/wiki_embedding.csv", index=False)

    def save_encoder(self):
        """
        [summary] : 학습한 encoder의 state_dict를 지정한 폴더에 저장
        """
        if self.num_neg != 0:
            p_encoder_path = os.path.join(self.save_dir, "passage")
            q_encoder_path = os.path.join(self.save_dir, "query")

            os.makedirs(p_encoder_path, exist_ok=True)
            os.makedirs(q_encoder_path, exist_ok=True)

            self.p_encoder.save_pretrained(p_encoder_path)
            self.q_encoder.save_pretrained(q_encoder_path)
        else:
            p_encoder_path = os.path.join(self.save_dir, "passage_in_batch")
            q_encoder_path = os.path.join(self.save_dir, "query_in_batch")

            os.makedirs(p_encoder_path, exist_ok=True)
            os.makedirs(q_encoder_path, exist_ok=True)

            self.p_encoder.save_pretrained(p_encoder_path)
            self.q_encoder.save_pretrained(q_encoder_path)

    def load_encoder(self):
        """
        [summary] : 학습된 encoder가 존재한다며 load를 하고 존재하지 않으면 train 시킨다.
        """
        if self.num_neg != 0:
            if os.path.isdir(os.path.join(self.save_dir, "passage")) and os.path.isdir(
                os.path.join(self.save_dir, "query")
            ):
                print("loading encoder !!!")
                self.p_encoder = self.p_encoder.from_pretrained(
                    os.path.join(self.save_dir, "passage")
                )
                self.q_encoder = self.q_encoder.from_pretrained(
                    os.path.join(self.save_dir, "query")
                )
            else:
                self.train()
        else:
            if os.path.isdir(
                os.path.join(self.save_dir, "passage_in_batch")
            ) and os.path.isdir(os.path.join(self.save_dir, "query_in_batch")):
                print("loading in_batch encoder !!!")
                self.p_encoder = self.p_encoder.from_pretrained(
                    os.path.join(self.save_dir, "passage_in_batch")
                )
                self.q_encoder = self.q_encoder.from_pretrained(
                    os.path.join(self.save_dir, "query_in_batch")
                )
            else:
                self.train_in_batch()

    def train(self, args=None):
        print("training by negative sampling data !!!")
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
        t_total = (
            len(self.train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

        for _ in train_iterator:
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(batch_size).long().to(args.device)

                    p_inputs = {
                        "input_ids": batch[0]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(args.device),
                        "attention_mask": batch[1]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[2].to(args.device),
                        "attention_mask": batch[3].to(args.device),
                    }

                    p_outputs = self.p_encoder(**p_inputs)
                    q_outputs = self.q_encoder(**q_inputs)

                    p_outputs = p_outputs.view(batch_size, -1, self.num_neg + 1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{loss.item():.3f}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        # self.save_encoder()
        torch.save(p_encoder.state_dict(), "./encoders/p_encoder.pt")
        torch.save(q_encoder.state_dict(), "./encoders/q_encoder.pt")

    def train_in_batch(self, args=None):
        print("training by in_batch negative sampling data !!!")
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
        t_total = (
            len(self.train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

        for _ in train_iterator:
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = (
                        torch.arange(0, args.per_device_train_batch_size)
                        .long()
                        .to(args.device)
                    )

                    p_inputs = {
                        "input_ids": batch[0].to(args.device),
                        "attention_mask": batch[1].to(args.device),
                        "token_type_ids": batch[2].to(args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device),
                    }

                    p_outputs = self.p_encoder(**p_inputs)
                    q_outputs = self.q_encoder(**q_inputs)

                    sim_scores = torch.matmul(
                        q_outputs, torch.transpose(p_outputs, 0, 1)
                    )

                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)

                    tepoch.set_postfix(loss=f"{loss.item():.3f}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        self.save_encoder()
        # torch.save(p_encoder.state_dict(), "./encoders/p_encoder_in_batch.pt")
        # torch.save(q_encoder.state_dict(), "./encoders/q_encoder_in_batch.pt")

    def get_relevant_doc(
        self,
        query: str,
        k: Optional[int] = 1,
        args=None,
        p_encoder=None,
        q_encoder=None,
    ) -> Tuple[List, List]:
        """[summary] : query가 1개 일 때 top-k개의 passage를 선택한다.
        Args:
            qurey : 1개의 질문
            k : 선택할 passage의 개수
        """
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder.to(args.device)

        if q_encoder is None:
            q_encoder = self.q_encoder.to(args.device)

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seq = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                return_tensors="pt",
                # return_token_type_ids=False,
            ).to(args.device)
            q_emb = q_encoder(**q_seq).to("cpu")

            p_embs = []

            for batch in tqdm(self.valid_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                p_emb = p_encoder(**p_inputs).to("cpu")
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(
            (len(self.valid_dataloader.dataset), -1)
        )

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return dot_prod_scores.squeeze(), rank[:k]

    def get_relevant_doc_bulk(
        self,
        queries: List,
        k: Optional[int] = 1,
        args=None,
        p_encoder=None,
        q_encoder=None,
    ) -> Tuple[List, List]:
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(args.device)
            q_embs = q_encoder(**q_seqs).to("cpu")  # (num_queries, emb_dim)

            p_embs = []

            for batch in tqdm(self.valid_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

                p_emb = p_encoder(**p_inputs).to("cpu")
                p_embs.append(p_emb)
        p_embs = torch.stack(p_embs, dim=0).view(len(self.valid_dataloader.dataset), -1)

        doc_scores = []
        doc_indices = []
        for i in range(len(queries)):
            dot_pord_scores = torch.matmul(
                q_embs[i].unsqueeze(0), torch.transpose(p_embs, 0, 1)
            )  # (1, num_passage)
            rank = torch.argsort(dot_pord_scores, dim=1, descending=True).squeeze()
            doc_scores.append(dot_pord_scores.squeeze()[:k])
            doc_indices.append(rank[:k])

        return doc_scores, doc_indices


# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument(
#         "--dataset_name", metavar="./data/train_dataset", type=str, help=""
#     )
#     parser.add_argument(
#         "--model_name_or_path",
#         metavar="bert-base-multilingual-cased",
#         type=str,
#         help="",
#     )
#     parser.add_argument("--data_path", metavar="./data", type=str, help="")
#     parser.add_argument(
#         "--context_path", metavar="wikipedia_documents", type=str, help=""
#     )
#     parser.add_argument("--use_faiss", metavar=False, type=bool, help="")

#     args = parser.parse_args()

#     # Test sparse
#     org_dataset = load_from_disk(args.dataset_name)
# full_ds = concatenate_datasets(
#     [
#         org_dataset["train"].flatten_indices(),
#         org_dataset["validation"].flatten_indices(),
#     ]
# )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
#     print("*" * 40, "query dataset", "*" * 40)
#     print(full_ds)

#     from transformers import AutoTokenizer

#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name_or_path,
#         use_fast=False,
#     )

#     retriever = SparseRetrieval(
#         tokenize_fn=tokenizer.tokenize,
#         data_path=args.data_path,
#         context_path=args.context_path,
#     )

#     query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

#     if args.use_faiss:

#         # test single query
#         with timer("single query by faiss"):
#             scores, indices = retriever.retrieve_faiss(query)

#         # test bulk
#         with timer("bulk query by exhaustive search"):
#             df = retriever.retrieve_faiss(full_ds)
#             df["correct"] = df["original_context"] == df["context"]

#             print("correct retrieval result by faiss", df["correct"].sum() / len(df))

#     else:
#         with timer("bulk query by exhaustive search"):
#             df = retriever.retrieve(full_ds)
#             df["correct"] = df["original_context"] == df["context"]
#             print(
#                 "correct retrieval result by exhaustive search",
#                 df["correct"].sum() / len(df),
#             )

#         with timer("single query by exhaustive search"):
#             scores, indices = retriever.retrieve(query)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")
    parser.add_argument("--save_dir", metavar="./encoders", type=str, help="")
    parser.add_argument("--num_neg", default=0, type=int, help="")

    args = parser.parse_args()

    set_seed(42)

    dense_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    model_checkpoint = args.model_name_or_path  # roberta-base

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(dense_args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(dense_args.device)

    dense_retriever = DenseRetrieval(
        args=dense_args,
        data_path=args.dataset_name,
        num_neg=args.num_neg,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        save_dir=args.save_dir,
    )

    dense_retriever.load_encoder()
    # dense_retriever.make_wiki_embedding(64)

    # query = "태어난 지 얼마 안 된 웨이게오쿠스쿠스는 무엇이 달라지는가?"

    # scores, results = dense_retriever.get_relevant_doc(query, k=5)

    # print(f"[Search Query] {query}\n")

    # indices = results.tolist()
    # for i, idx in enumerate(indices):
    #     print(f"\nTop-{i + 1}th Passage (Index {idx}, Score : {scores[idx]})")
    #     print(dense_retriever.p_dataset[idx])