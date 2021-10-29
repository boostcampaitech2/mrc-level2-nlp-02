import pandas as pd
import os
import pickle
from typing import Optional

from datasets import load_from_disk
from transformers import AutoTokenizer, HfArgumentParser

from rt_bm25 import SparseRetrieval
from rt_arguments import (
    EncoderModelArguments,
    RtDataTrainingArguments,
)


def make_custom_dataset_with_bm25(
    dataset: pd.DataFrame, save_path: str, top_k: int = 1, pt_num: Optional[str] = None
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

    file_name = os.path.join(save_path, f"bm25_top{top_k}_pp{pt_num}.pickle")

    with open(file_name, "wb") as f:
        pickle.dump(custom_dataset, f)


def main(model_args, data_args):
    datasets = load_from_disk(data_args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=data_args.data_path,
        context_path=data_args.context_path,
        pt_num=data_args.preprocessing_pattern,
    )

    if data_args.preprocessing_pattern == None:
        data_args.preprocessing_pattern = 0

    bm25_path = os.path.join(
        data_args.save_dir,
        f"bm25_top{data_args.top_k_retrieval}_pp{data_args.preprocessing_pattern}.csv",
    )

    retriever.get_sparse_BM25()

    if os.path.isfile(bm25_path):
        df = pd.read_csv(bm25_path)
    else:
        df = retriever.retrieve_BM25(
            datasets["train"],
            topk=data_args.top_k_retrieval,
            score_ratio=data_args.score_ratio,
        )
        df.to_csv(bm25_path, index=False)

    make_custom_dataset_with_bm25(
        df, data_args.pickle_save_dir, top_k=data_args.top_k_retrieval, pt_num=data_args.preprocessing_pattern
    )


if __name__ == "__main__":
    parser = HfArgumentParser((EncoderModelArguments, RtDataTrainingArguments))

    encoder_args, rt_data_args = parser.parse_args_into_dataclasses()

    main(encoder_args, rt_data_args)
