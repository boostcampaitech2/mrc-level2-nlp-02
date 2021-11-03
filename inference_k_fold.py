"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
import torch
from typing import Callable, List, Dict, NoReturn, Tuple

import numpy as np

from datasets import (
    load_metric,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer

from retriever import retriever_sparse_BM25
from retriever import retriever_sparse_ES
from retriever.retriever_dense import DenseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    LoggingArguments,
)

import wandb
from dotenv import load_dotenv
import os

from retriever.model_encoder import BertEncoder


from preprocessor import Preprocessor
import pandas as pd

logger = logging.getLogger(__name__)

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    
    # dataclass를 통해 변수를 만들고 HfArgumentParser를 통해 합쳐서 사용
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, log_args = parser.parse_args_into_dataclasses()
    
    #trainingarguments
    training_args.per_device_eval_batch_size = 8
    
    #wandb
    load_dotenv(dotenv_path=log_args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)
    
    wandb.init(
        entity="klue-level2-nlp-02",
        project=log_args.project_name,
        name=log_args.wandb_name + "_eval" if training_args.do_eval==True else "_inference",
        group=model_args.model_name_or_path,
    )
    wandb.config.update(training_args)


    # training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #데이터셋을 불러옵니다.
    datasets = load_from_disk(data_args.dataset_name)

    #cache 파일을 정리합니다.
    datasets.cleanup_cache_files()
    
    if training_args.do_predict==True and data_args.add_special_tokens_query_flag:
        q_type_data = pd.read_csv("./csv/question_tag_testset.csv",index_col=0)
        train_data = datasets['validation'].to_pandas()
        train_data['question'] = train_data['question']+' '+q_type_data['Q_tag']
        datasets['validation'] = datasets['validation'].from_pandas(train_data)
        print(datasets['validation']['question'][0])
        print("======================================= predict Tag complete============================")
        
    if training_args.do_eval==True and data_args.preprocessing_pattern != None:
        if data_args.add_special_tokens_query_flag:
            q_type_data = pd.read_csv("./csv/question_tag_validset.csv",index_col=0)
            
            train_data = datasets['validation'].to_pandas()
            train_data['question']=train_data['question']+' '+q_type_data['Q_tag']
            datasets['validation'] = datasets['validation'].from_pandas(train_data)
            print(datasets['validation']['question'][0])
            print("======================================= Tag complete============================")

        datasets = Preprocessor.preprocessing(data = datasets, pt_num=data_args.preprocessing_pattern)
    print(datasets)
    
    model_path = model_args.model_name_or_path

    try : 
        if '0' in os.listdir(model_args.model_name_or_path) and '1' in os.listdir(model_args.model_name_or_path) :
            model_path = os.path.join(model_args.model_name_or_path, '0')
    except :
        model_path = model_args.model_name_or_path
    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    try :
        if '0' in os.listdir(model_args.model_name_or_path) and '1' in os.listdir(model_args.model_name_or_path) :
            model_path_list = os.listdir(model_args.model_name_or_path)
            sub_path = os.path.join(model_args.model_name_or_path, model_path_list[0])
            model = AutoModelForQuestionAnswering.from_pretrained(
                sub_path, config=config
                )
            model_sd = model.state_dict()
            for sub_path in model_path_list[1:] :
                sub_path = os.path.join(model_args.model_name_or_path, sub_path)
                sub_model = AutoModelForQuestionAnswering.from_pretrained(
                    sub_path, config=config
                    )
                sub_model_sd = sub_model.state_dict()
                for layer in model_sd :
                    model_sd[layer] = (model_sd[layer] + sub_model_sd[layer])
            for key in model_sd :
                model_sd[key] = model_sd[key] / float(len(model_path_list))
            model.load_state_dict(model_sd)
    except :
        pass

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval == "sparse":
        datasets = run_sparse_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )
    elif data_args.eval_retrieval == "elastic_sparse":
        datasets = retriever_sparse_ES.run_elastic_sparse_retrieval(
            datasets,
            training_args,
            data_args,
        )
    elif data_args.eval_retrieval == "dense":
        datasets = run_dense_retrieval(
            "klue/bert-base", datasets, training_args, data_args
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

def run_dense_retrieval(
    model_checkpoint: str,
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
):
    dense_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(training_args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(training_args.device)
    if data_args.num_neg != 0:
        print("change batch size default(8) to 2 !!!")
        training_args.per_device_train_batch_size = 2
    retriever = DenseRetrieval(
        training_args,
        data_path="/opt/ml/data/train_dataset",
        num_neg=data_args.num_neg,
        tokenizer=dense_tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        save_dir="./encoders",
    )

    retriever.load_encoder()
    df = retriever.retrieve(
        datasets["validation"],
        "/opt/ml/data/wiki_embedding.csv",
        topk=data_args.top_k_retrieval,
    )

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
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
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = '../data',
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    # retriever 설정
    retriever = retriever_sparse_BM25.SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path,
        pt_num=data_args.preprocessing_pattern
    )
    
    # Passage Embedding 만들기
    retriever.get_sparse_BM25()
    df = retriever.retrieve_BM25(datasets['validation'], topk=data_args.top_k_retrieval, score_ratio=data_args.score_ratio)
    
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features( # Features로 데이터 셋 형식화?
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features( # Features로 형식화?
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
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")
    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()