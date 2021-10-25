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

from model_encoder import RobertaEncoder, BertEncoder

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


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

        with open(os.path.join('/opt/ml/data/wikipedia_documents.json'), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        wiki_contexts = list(
        dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(wiki_contexts)}")
        self.wiki_texts = list(wiki_contexts)

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

    def make_wiki_embedding(self, args=None, batch_size: int = 64):
        """
        [summary] : 학습시킨 p_encoder를 이용해 wiki text의 embedding vector를 생성하고 csv 파일로 저장합니다.
        """
        if args is None:
            args = self.args
        print(f"wiki text size : {len(self.wiki_texts)}")

        wiki_seqs = self.tokenizer(
            self.wiki_texts, padding="max_length", truncation=True, return_tensors="pt"
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

    def retrieve(
        self,
        queries: Dataset,
        wiki_embedding_path: str,
        topk: Optional[int] = 1,
    ) -> pd.DataFrame:
        """
        [summary] : 입력받은 쿼리와 wiki embedding vector의 dot product score를 구하고 topk개의 passage를 선택해서 출력
        wiki_embedding : wiki embedding vector가 저장되어있는 파일의 path
        """
        print("start dense retrieve!!!")

        if wiki_embedding_path is None :
            df = pd.read_csv("/opt/ml/data/wiki_embedding.csv")
        else :
            df = pd.read_csv(wiki_embedding_path)

        with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(queries['question'], df, k=topk)
        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        for idx, example in enumerate(tqdm(queries, desc="Dense retrieval: ")):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": doc_indices[idx],
                "context": " ".join([self.wiki_texts[pid] for pid in doc_indices[idx]]),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)
        cqas = pd.DataFrame(total)
        return cqas

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

                    p_outputs = p_outputs.view(
                        batch_size, self.num_neg + 1, -1
                    ).transpose(1, 2)
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

        self.save_encoder()

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
        queries: Dataset,
        df: pd.DataFrame,
        k: Optional[int] = 1,
        args=None,
    ) -> Tuple[List, List]:
        
        if args is None:
            args = self.args

        p_embs = torch.Tensor(df.values)

        print(f"passage size : {df.shape}")

        self.q_encoder.to("cuda")
        
        with torch.no_grad():
            q_seqs = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to("cuda")
            q_embs = self.q_encoder(**q_seqs).to("cpu")

        sim_scores = torch.matmul(q_embs, p_embs.T)  # (quries, wiki_data_len)

        doc_scores = []
        doc_indices = []

        for i in range(len(queries)):
            rank = torch.argsort(sim_scores[i], dim=0, descending=True)
            doc_scores.append(sim_scores[i][rank[:k]])
            doc_indices.append(rank[:k])
        return doc_scores, doc_indices

if __name__ == "__main__":
    model_name_or_path="klue/bert-base"
    data_path="/opt/ml/data"
    context_path="wikipedia_documents.json"

    dense_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    import argparse
    args = argparse.Namespace(
        dataset_name="../data",
        model_name_or_path="klue/bert-base",
        data_path="/opt/ml/data",
        context_path="wikipedia_documents.json",
        save_dir="./encoders",
        num_neg=0,  
    )


    p_encoder = BertEncoder.from_pretrained(model_name_or_path).to(dense_args.device)
    q_encoder = BertEncoder.from_pretrained(model_name_or_path).to(dense_args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    datasets = load_from_disk('/opt/ml/data/train_dataset') 

    klue_dense= DenseRetrieval(
        args=dense_args,
        data_path="../data/train_dataset",
        num_neg=args.num_neg,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        save_dir=args.save_dir,
    )

    klue_dense.load_encoder()
    klue_dense.make_wiki_embedding(64)    
    # wiki_embedding = pd.read_csv('/opt/ml/data/wiki_embedding.csv')
    # max_K = 10
    # klue_dense.get_relevant_doc_bulk(datasets['validation'], wiki_embedding, k=max_K)
    # df = retriever.retrieve(datasets["validation"], topk=5)
    # breakpoint()