import os
import pickle
from typing import List
from datasets.arrow_dataset import Dataset
from datasets.load import load_from_disk
from tqdm.auto import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    set_seed,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
)
from datasets import (
    load_from_disk,
    Value,
    Features,
    Dataset,
    DatasetDict
)

from rt_arguments import (
    EncoderModelArguments,
    RtDataTrainingArguments,
)
from rt_model import klueRobertaEncoder
from rt_bm25 import SparseRetrieval

def load_encoder(model_name, training_args, data_args, p_encoder, q_encoder):
    # 파일 이름 model_epoch_batch_topk_acs_pp(preprocessing pattern)
    file_name = f"{model_name.split('/')[-1]}_ep{int(training_args.num_train_epochs)}_bs{training_args.per_device_train_batch_size}_topk{data_args.top_k_retrieval}_acs{training_args.gradient_accumulation_steps}_pp{data_args.preprocessing_pattern}"
    # test 용
    # file_name = f"ep{training_args.num_train_epochs}_bs{training_args.per_device_train_batch_size}_topk{data_args.top_k_retrieval}_acs{training_args.gradient_accumulation_steps}"
    full_path = os.path.join(training_args.output_dir, file_name)
    p_encoder_path = os.path.join(full_path, "passage.pt")
    q_encoder_path = os.path.join(full_path, "query.pt")

    assert os.path.isfile(p_encoder_path) or os.path.isfile(q_encoder_path), "rt_train을 실행해서 model parameter를 저장해야 합니다."

    p_encoder.load_state_dict(torch.load(p_encoder_path))
    q_encoder.load_state_dict(torch.load(q_encoder_path))

    print("finish load model state dict !!!")

    return p_encoder, q_encoder

def loading_prepro_wiki(data_args, tokenizer):

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=data_args.data_path,
        context_path=data_args.context_path,
        pt_num=data_args.preprocessing_pattern,
    )
    
    return retriever.contexts

def save_and_load_wiki_embedding(model_name, training_args, data_args, p_encoder, wiki_texts, tokenizer):
    """
    [summary] : 학습시킨 p_encoder를 이용해 wiki text의 embedding vector를 생성하고 csv 파일로 저장합니다.
    """
    file_name = f"{model_name.split('/')[-1]}_ep{int(training_args.num_train_epochs)}_bs{training_args.per_device_train_batch_size}_topk{data_args.top_k_retrieval}_acs{training_args.gradient_accumulation_steps}_pp{data_args.preprocessing_pattern}.csv"

    os.makedirs("/opt/ml/data/wiki", exist_ok=True)

    print(f"wiki text size : {len(wiki_texts)}")

    if os.path.isfile(os.path.join("/opt/ml/data/wiki", file_name)):
        emb_df = pd.read_csv(os.path.join("/opt/ml/data/wiki", file_name))
        return emb_df
    else :
        wiki_seqs = tokenizer(
        wiki_texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(training_args.device)

        wiki_dataset = TensorDataset(
            wiki_seqs["input_ids"],
            wiki_seqs["attention_mask"],
            wiki_seqs["token_type_ids"],
        )

        wiki_dataloader = DataLoader(wiki_dataset, batch_size=training_args.per_device_eval_batch_size)

        p_encoder.to(training_args.device)

        with torch.no_grad():
            p_encoder.eval()

            p_embs = []

            for batch in tqdm(wiki_dataloader, unit="batch"):
                batch = tuple(t.to(training_args.device) for t in batch)
                wiki_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                p_emb = p_encoder(**wiki_inputs).to("cpu")
                p_embs.append(p_emb)

        p_embs = torch.cat(p_embs, dim=0).numpy()
        print(f"wiki passage embedding shape:{p_embs.shape}")
        print("\nMake wiki embedding csv file !!!")

        emb_df = pd.DataFrame(p_embs)
        emb_df.to_csv(os.path.join("/opt/ml/data/wiki", file_name), index=False)
        
        return emb_df

def retrieve(q_encoder, tokenizer, queries: Dataset, wiki_texts : List, wiki_embedding : pd.DataFrame, topk : int):
    print("start dense retrieve!!!")

    p_embs = torch.Tensor(wiki_embedding.values)

    print(f"passage size : {p_embs.shape}")

    q_encoder.to("cuda")

    with torch.no_grad():
        q_seqs = tokenizer(
            queries["question"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to("cuda")
        q_embs = q_encoder(**q_seqs).to("cpu")

    sim_scores = torch.matmul(q_embs, p_embs.T) # (quries, wiki_data_len)
    
    print(f"sim_score size : {sim_scores.shape}")  

    doc_scores = []
    doc_indices = []

    for i in range(len(queries)):
        rank = torch.argsort(sim_scores[i], dim=0, descending=True)
        doc_scores.append(sim_scores[i][rank[:topk]])
        doc_indices.append(rank[:topk])

    # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
    total = []
    for idx, example in enumerate(tqdm(queries, desc="Dense retrieval: ")):
        tmp = {
            # Query와 해당 id를 반환합니다.
            "question": example["question"],
            "id": example["id"],
            # Retrieve한 Passage의 id, context를 반환합니다.
            "context_id": doc_indices[idx],
            "context": " ".join([wiki_texts[pid] for pid in doc_indices[idx]]),
        }
        if "context" in example.keys() and "answers" in example.keys():
            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
            tmp["original_context"] = example["context"]
            tmp["answers"] = example["answers"]
        total.append(tmp)
    top_k_passage = pd.DataFrame(total)
    return top_k_passage


def main(model_args, data_args, training_args):
    print(f"Inference !!!")
    print(f"model is from {encoder_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )

    p_encoder = klueRobertaEncoder(model_args.model_name_or_path)
    q_encoder = klueRobertaEncoder(model_args.model_name_or_path)

    p_encoder, q_encoder = load_encoder(encoder_args.model_name_or_path, training_args, data_args, p_encoder, q_encoder)

    wiki_texts = loading_prepro_wiki(data_args, tokenizer)

    wiki_embedding_df = save_and_load_wiki_embedding(model_args.model_name_or_path, training_args, data_args, p_encoder, wiki_texts, tokenizer)

    datasets = load_from_disk(data_args.dataset_name)

    top_k_df = retrieve(q_encoder, tokenizer, datasets, wiki_texts, wiki_embedding_df, data_args.DRP_top_k)

    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(top_k_df, features=f)})

    path = f"{model_args.model_name.split('/')[-1]}_ep{int(training_args.num_train_epochs)}_bs{training_args.per_device_train_batch_size}_topk{data_args.DRP_top_k}_acs{training_args.gradient_accumulation_steps}_pp{data_args.preprocessing_pattern}"
    
    folder_name = os.path.join("opt/ml/data/top_k", path)
    os.makedirs(folder_name, exist_ok=True)

    with open(os.path.join(folder_name, "DPR_datasets.pickle"), "wb") as f:
        pickle.dump(datasets, f)

if __name__ == "__main__":
    parser = HfArgumentParser(
        (EncoderModelArguments, RtDataTrainingArguments, TrainingArguments)
    )

    encoder_args, rt_data_args, training_args = parser.parse_args_into_dataclasses()

    main(encoder_args, rt_data_args, training_args)
    
    print("Complete inference !!!")