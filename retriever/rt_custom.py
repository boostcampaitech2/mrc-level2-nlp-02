import pandas as pd
import os
import pickle
from typing import Optional
import re

from datasets import load_from_disk
from transformers import AutoTokenizer, HfArgumentParser

from rt_bm25 import SparseRetrieval
from rt_arguments import (
    EncoderModelArguments,
    RtDataTrainingArguments,
)

pattern_dict={
                "1" : re.compile("(\\n)+|(\\\\n)+|(\\xa0)|(\\u3000)"),
                "2" : re.compile("(\\\\n)+|(\\n)+|[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥()?!∧≪≫『』\'<>〈〉:「」＜＞<>》《・\"-“”\s\.\‘’%,]"),
                "3" : re.compile('['+chr(0)+'-'+chr(31)+chr(8191)+'-'+chr(12288)+chr(55204)+'-'+chr(63743)+']')}
common = re.compile('(\s+)')

def make_custom_dataset_with_bm25(
    dataset: pd.DataFrame, save_path: str, top_k: int = 1, pt_num: Optional[str] = None
):

    os.makedirs(save_path, exist_ok=True)

    custom_dataset = {}

    answers = dataset.answers.tolist()
    question = dataset.question.tolist()
    original_context = dataset.original_context.tolist()

    pd_data = pd.DataFrame({"original_context" : original_context})
    if pt_num is not None :
        for num in pt_num:
            preprocessing = lambda x : pattern_dict[num].sub(" ", x)
            blank_remove = lambda x : common.sub(" ", x)
            pd_data["original_context"] = pd_data.original_context.map(preprocessing)
            pd_data["original_context"] = pd_data.original_context.map(blank_remove)
        original_context = pd_data.original_context.to_list()

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

    os.makedirs(data_args.save_dir, exist_ok=True)

    bm25_path_train = os.path.join(
        data_args.save_dir,
        f"bm25_top{data_args.top_k_retrieval}_pp{data_args.preprocessing_pattern}.csv",
    )
    bm25_path_valid = os.path.join(
        data_args.save_dir,
        f"bm25_top{data_args.top_k_retrieval}_pp{data_args.preprocessing_pattern}_val.csv",
    )

    retriever.get_sparse_BM25()

    if os.path.isfile(bm25_path_train) and os.path.isfile(bm25_path_valid):
        df_train = pd.read_csv(bm25_path_train)
        df_valid = pd.read_csv(bm25_path_valid)
    else:
        df_train = retriever.retrieve_BM25(
            datasets["train"],
            topk=data_args.top_k_retrieval,
            score_ratio=data_args.score_ratio,
        )
        df_valid = retriever.retrieve_BM25(
            datasets["validation"],
            topk=data_args.top_k_retrieval,
            score_ratio=data_args.score_ratio,
        )
        df_train.to_csv(bm25_path_train, index=False)
        df_valid.to_csv(bm25_path_valid, index=False)

    make_custom_dataset_with_bm25(
        df_train, data_args.train_pickle_save_dir, top_k=data_args.top_k_retrieval, pt_num=data_args.preprocessing_pattern
    )
    make_custom_dataset_with_bm25(
        df_valid, data_args.valid_pickle_save_dir, top_k=data_args.top_k_retrieval, pt_num=data_args.preprocessing_pattern
    )


if __name__ == "__main__":
    parser = HfArgumentParser((EncoderModelArguments, RtDataTrainingArguments))

    encoder_args, rt_data_args = parser.parse_args_into_dataclasses()

    main(encoder_args, rt_data_args)
