import logging
import os
import sys
import math
import torch
import pandas as pd

from typing import NoReturn
from datasets import load_metric, load_from_disk, Dataset, DatasetDict

from transformers import AutoConfig, AutoModelForQuestionAnswering
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    LoggingArguments,
)

from custom_tokenizer import load_pretrained_tokenizer
from dotenv import load_dotenv
from preprocessor import Preprocessor
from sklearn.model_selection import KFold
import wandb

logger = logging.getLogger(__name__)

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # dataclass를 통해 변수를 만들고 HfArgumentParser를 통해 합쳐서 사용합니다.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LoggingArguments, TrainingArguments)
    )

    model_args, data_args, log_args, training_args = parser.parse_args_into_dataclasses()
    
    #wandb
    load_dotenv(dotenv_path=log_args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    wandb.init(
        entity="klue-level2-nlp-02",
        project=log_args.project_name,
        nname=log_args.wandb_name + "_train/train" if training_args.do_train==True else log_args.wandb_name + "_train/eval",
        group=model_args.model_name_or_path,
    )
    wandb.config.update(training_args)

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 데이터셋을 불러옵니다.
    datasets = load_from_disk(data_args.dataset_name)

    model_path = model_args.model_name_or_path
    # 이전에 K-fold training을 진행했다면 0~K 이름의 directory가 존재합니다.
    # tokenizer와 config설정을 위해 0번째 directory path를 지정합니다.
    try : 
        if '0' in os.listdir(model_args.model_name_or_path) and '1' in os.listdir(model_args.model_name_or_path) :
            model_path = os.path.join(model_args.model_name_or_path, '0')
    except :
        model_path = model_args.model_name_or_path
    
    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_path)
    print(config)

    # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
    # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
    # rust version이 비교적 속도가 빠릅니다.
    tokenizer = load_pretrained_tokenizer(
            pretrained_model_name_or_path = model_args.model_name_or_path,
            data_selected = data_args.data_selected,
            datasets=datasets,
            add_special_tokens_flag = data_args.add_special_tokens_flag or data_args.add_special_tokens_query_flag,
            use_fast=True)
    
    # 추가된 vocab size를 확인합니다.
    print("\n","num of added vocab in tokenizer : ", len(tokenizer.vocab) - config.vocab_size)
    
    # Question tag를 붙입니다.(ex. 나는 언제 밥을 먹을까?[WHEN])
    if data_args.add_special_tokens_query_flag:
        # do_train 시, train에 관한 데이터셋에 Question tag 붙입니다.
        if training_args.do_train:
            q_type_data = pd.read_csv("./csv/question_tag_trainset.csv",index_col=0)
            data_type = "train"
        # do_eval 시, validation에 관한 데이터셋에 Question tag 붙입니다.
        elif training_args.do_eval:
            q_type_data = pd.read_csv("./csv/question_tag_validset.csv",index_col=0)
            data_type = "validation"
        train_data = datasets[data_type].to_pandas()
        train_data['question']=train_data['question']+' '+q_type_data['Q_tag']
        datasets[data_type] = datasets[data_type].from_pandas(train_data)
        print(" "+"*"*50,"\n","*"*50,"\n","*"*50)
        print(" ***** question tag 끝!: ", datasets[data_type]['question'][0],"******")
        print(" "+"*"*50,"\n","*"*50,"\n","*"*50,"\n\n")

    # rtt 데이터셋이 존재할 경우 기존 데이터셋과 합칩니다.
    if data_args.rtt_dataset_name != None and training_args.do_train:
        print(" "+"*"*50,"\n","*"*50,"\n","*"*50)
        print(" ***** rtt 데이터 병합 전 데이터 개수: ", len(datasets['train']),"******")
        rtt_data = pd.read_csv(data_args.rtt_dataset_name,index_col=0)
        
        if data_args.add_special_tokens_query_flag:
            q_data = pd.read_csv("./csv/question_tag_rtt_papago_ner.csv",index_col=0)
            rtt_data['question']=rtt_data['question']+' '+q_data['Q_tag']
            print(" ***** rtt question tag 끝!: ", rtt_data.loc[0]['question'],"******")
        rtt_data['answers'] = rtt_data.answers.map(eval)

        train_data = datasets['train'].to_pandas()
        new_data = pd.concat([train_data,rtt_data])
        new_data = new_data.drop_duplicates(subset="question").reset_index(drop=True)
        datasets['train'] = datasets['train'].from_pandas(new_data)
        print(" "+"*"*50,"\n","*"*50,"\n","*"*50)
        print(" ***** rtt 데이터 병합 후 데이터 개수: ", len(datasets['train']),"******")
        print(" "+"*"*50,"\n","*"*50,"\n","*"*50,"\n\n")
    print(datasets)
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path), # Load the model weights from a TensorFlow checkpoint save file
        config=config,
    )

    # 이전에 K-fold training을 진행했다면 0~K 이름의 directory가 존재합니다.
    try :
        if '0' in os.listdir(model_args.model_name_or_path) and '1' in os.listdir(model_args.model_name_or_path) :
            model_path_list = os.listdir(model_args.model_name_or_path)
            sub_path = os.path.join(model_args.model_name_or_path, model_path_list[0])
            model = AutoModelForQuestionAnswering.from_pretrained(
                sub_path, config=config
                )
            # Model weight average를 위해 for loop를 돌며 weight를 평균 내어줍니다.
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

    # vocab size가 추가되었을 때를 위해 model resize를 진행합니다.
    model.resize_token_embeddings(len(tokenizer))
    assert model.vocab_size == len(tokenizer), "embedding size and vocab size is not equal"
    print("\n",f"embedding size and vocab size is equal \n [model vocab_size] {model.vocab_size} || [tokenizer vocab_size] {len(tokenizer)}" )

    #cache 파일을 정리합니다.
    datasets.cleanup_cache_files()
    
    print(data_args.preprocessing_pattern)
    print("\n","전처리 전: \n",datasets['train'][0])

    # 원하는 전처리를 수행합니다.
    if data_args.preprocessing_pattern != None:
        datasets = Preprocessor.preprocessing(data = datasets, pt_num = data_args.preprocessing_pattern)
        print("\n","전처리 후: \n",datasets['train']['context'][0])
    
    output_dir = training_args.output_dir
    if training_args.do_train :
        cv = KFold(n_splits=model_args.k_fold, random_state=training_args.seed,shuffle=True)
        for i, (t, v) in enumerate(cv.split(datasets['train'])) :
            kf_datasets = DatasetDict()
            kf_datasets['train'] = datasets['train'].select(t.tolist())
            kf_datasets['validation'] = datasets['validation']
            
            training_args.output_dir = os.path.join(output_dir, str(i))
            print(training_args.output_dir)
                        
            print(
                type(training_args),
                type(model_args),
                type(datasets),
                type(tokenizer),
                type(model),
            )
            
            run_mrc(data_args, training_args, model_args, kf_datasets, tokenizer, model)
            
    if training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    """
        Dataset을 전처리한 뒤 Reader model을 실행
    """
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right" # right 시, question|context!

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Train preprocessing / 전처리를 진행합니다.
    def prepare_train_features(examples):
        """
            truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
            각 example들은 이전의 context와 조금씩 겹치게됩니다.
        """
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first", # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True, # 길이를 넘어가는 토큰들을 반환할 것인지
            return_offsets_mapping=True,  # 각 토큰에 대해 (char_start, char_end) 정보를 반환한 것인지
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index = 0

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                # right면, question(0)이 먼저 오므로 1
                # left면, context(1)이 먼저 오므로 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0): # context 시작점
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0): # context 끝나는 점
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
        """
            truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
            각 example들은 이전의 context와 조금씩 겹치게됩니다.
        """
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

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    if training_args.do_eval:
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
    # fp16 -> Whether to use 16-bit (mixed) precision training instead of 32-bit training. (default: false)
    # pad_to_multiple_of -> padding한다는 의미?
    print("-------------------------------------")
    print("training_args.fp16 :", training_args.fp16) # False
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        """
            start logits과 end logits을 original context의 정답과 match시킵니다.
        """
        predictions = postprocess_qa_predictions(
            examples=examples, # 전처리 되지 않은 dataset
            features=features, # 전처리 된 dataset
            predictions=predictions, # start logits과 the end logits을 나타내는 two arrays
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        ## predictions으로 id와 예측 text가 나온다.
        
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

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        total_steps = math.ceil(len(train_dataset)*training_args.num_train_epochs/training_args.per_device_train_batch_size)
        trainer.create_optimizer_and_scheduler(total_steps, data_args.num_cycles, data_args.another_scheduler_flag)
        
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        #elif os.path.isdir(model_args.model_name_or_path):
        #    checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
