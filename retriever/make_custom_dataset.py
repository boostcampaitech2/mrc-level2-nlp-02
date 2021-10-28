from datasets import load_from_disk
from transformers import AutoTokenizer

import pandas as pd
import json
import os
import pickle

from rt_bm25 import SparseRetrievalBM25


import argparse


def make_custom_dataset_with_bm25(
    dataset: pd.DataFrame, save_path: str, top_k: int = 1
):
    custom_dataset = {}

    answers = dataset.answers.tolist()
    question = dataset.question.tolist()
    original_context = dataset.original_context.tolist()
    top_k_passage = []
    target = []

    for idx, passages in enumerate(dataset.context.tolist()):
        p_list = passages.split("▦")

        if original_context[idx] in p_list:
            top_k_passage.append(p_list)
            target_index = p_list.index(original_context[idx])
            target.append(target_index)
        else:
            p_list = [original_context[idx]] + p_list[:-1]
            top_k_passage.append(p_list)
            target.append(0)

    custom_dataset["answers"] = answers
    custom_dataset["question"] = question
    custom_dataset["original_context"] = original_context
    custom_dataset["top_k_passage"] = top_k_passage
    custom_dataset["target"] = target

    file_name = os.path.join(save_path, f"bm25_{top_k}.pickle")

    with open(file_name, "wb") as f:
        pickle.dump(custom_dataset, f)


def main(args):
    datasets = load_from_disk(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
    retriever = SparseRetrievalBM25(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    bm25_path = os.path.join(
        args.save_dir, f"bm25_top_{args.top_k_retrieval}_train.csv"
    )

    retriever.get_sparse_BM25()

    if os.path.isfile(bm25_path):
        df = pd.read_csv(bm25_path)
    else:
        df = retriever.retrieve_BM25(datasets["train"], topk=args.top_k_retrieval)
        df.to_csv(bm25_path, index=False)

    make_custom_dataset_with_bm25(df, args.pickle_save_dir, top_k=args.top_k_retrieval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="/opt/ml/data/train_dataset")
    parser.add_argument("--model_name_or_path", default="klue/bert-base")
    parser.add_argument("--data_path", default="/opt/ml/data/")
    parser.add_argument(
        "--context_path",
        default="wikipedia_documents.json",
    )
    parser.add_argument("--top_k_retrieval", default=1, type=int)
    parser.add_argument("--save_dir", default="/opt/ml/data/bm25")
    parser.add_argument("--pickle_save_dir", default="/opt/ml/data/pickle")

    args = parser.parse_args()

    main(args)
