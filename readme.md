# Pstage_02_MRC

Solution for MRC Competitions in 2nd BoostCamp AI Tech 2기 by **메타몽팀 (2조)**

## Content
- [Competition Abstract](#competition-abstract)
- [Result](#result)
- [Hardware](#hardware)
- [Operating System](#operating-system)
- [Archive Contents](#archive-contents)
- [Getting Started](#getting-started)
  * [Dependencies](#dependencies)
  * [Install Requirements](#install-requirements)
- [Arguments](#arguments)
  * [Model Arguments](#model-arguments)
  * [DataTrainingArguments](#datatrainingarguments)
  * [LoggingArguments](#loggingarguments)
- [Running Command](#running-command)
  * [Train](#train)
  * [Reader evaluation](#reader-evaluation)
  * [ODQA evaluation](#odqa-evaluation)
  * [Inference prediction](#inference-prediction)
  * [Soft-voting Ensemble](#soft-voting-ensemble)
  * [Hard-voting Ensemble](#hard-voting-ensemble)
- [Reference](#reference)


## Competition Abstract

- 주어지는 지문이 따로 존재하지 않을 때 사전에 구축되어 있는 대용량의 corpus에서 질문에 대답할 수 있는 문서를 찾고, 다양한 종류의 질문에 대답하는 인공지능 모델 개발
- 데이터셋 통계:
  - Corpus : Wikipedia 약 5,7000개 문서
  - train_data : 3,952개 (Context, Question, Answer)
  - validation_data : 240개 (Context, Question, Answer)
  - test_data : 600개 (Question)

## Result

|         |   EM   |   F1   | RANK |
|:-------:|:------:|:------:|:----:|
| Public  | 74.580 | 83.100 |  3   |
| Private | 70.280 | 79.530 |  5   |


## Hardware

- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Operating System

- Ubuntu 18.04.5 LTS

## Archive Contents

- mrc-level2-nlp-02 : 구현 코드와 모델 checkpoint 및 모델 결과를 포함하는 디렉토리

```
mrc-level2-nlp-02/
├── utils
│   ├── crawling_papago_rtt.ipynb
│   ├── nbest_ensemble.ipynb
│   ├── Question type Tagging.ipynb
│   ├── question_generation.ipynb
│   └── use_ner.ipynb
├── data
│   ├── papago_ner.csv
│   ├── question_generation.csv
│   ├── question_tag_rtt_papago_ner.csv
│   ├── question_tag_testset.csv
│   ├── question_tag_trainset.csv
│   ├── question_tag_validset.csv
│   ├── trainset_rtt_papago.csv
│   └── trainset_rtt_pororo.csv
├── arguments.py
├── custom_tokenizer.py
├── inference.py
├── inference_k_fold.py
├── preprocessor.py
├── rt_bm25.py
├── train.py
├── train_k_fold.py
├── trainer_qa.py
└── utils_qa.py
```
- `utils/` : 해당 디렉토리 내 ipynb 파일 실행 시 data 디렉토리에 csv 파일 생성
- `data/` : train/inference 시 활용하는 데이터 파일
- `inference.py` : retriever-reader inference 후 predictions.json 및 nbest_predictions.json 생성
- `inference_k_fold.py` : k fold를 사용하여 inference하는 파일
- `preprocessor.py` : 데이터 전처리용 코드
- `rt_bm25.py` : bm25를 사용한 retriever
- `train.py` : reader 모델 학습을 위한 파일
- `train_k_fold.py` : reader 모델 학습시 k fold 적용 파일
- `trainer_qa.py` : Question Answering Trainer를 정의하는 파일
- `utils_qa.py` : Question Answering 후처리(post processing) 코드

## Getting Started

### Dependencies

- torch==1.6.0
- transformers==4.11.0
- datasets==1.4.0

### Install Requirements

```
sh requirement_install.sh
```

## Arguments
    
### Model Arguments

|      argument       | description                                                                                   | default                                      |
| :-----------------: | :-------------------------------------------------------------------------------------------- | :------------------------------------------- |
|      model_name_or_path      | 사용할 모델 선택                                                                           | klue/roberta-large                                |
|         rt_model_name       | 사용할 모델 선택                                                                 | klue/bert-base                               |
|   config_name    | Pretrained된 model config 경로                                                                             | klue/roberta-large |
|     tokenizer_name     | customized tokenizer 경로 선택                                                               | None                                        |
| customized_tokenizer_flag | customized roberta tokenizer 로드하기 | False|
| k_fold | K-fold validation의 k 선택 | 5|

### DataTrainingArguments

|      argument       | description                                                                                   | default                                      |
| :-----------------: | :-------------------------------------------------------------------------------------------- | :------------------------------------------- |
|      dataset_name      | 사용할 데이터셋 이름 지정                                                                           | /opt/ml/data/train_dataset                                |
|         overwrite_cache       |  캐시된 training과 evaluation set을 overwrite하기                                                                | False                               |
|preprocessing_num_workers|전처리동안 사용할 prcoess 수 지정|2|
|max_seq_length|Sequence 길이 지정|384|
|pad_to_max_length|max_seq_length에 모든 샘플 패딩할지 결정|True|
|doc_stride|얼마나 stride할지 결정|128|
|max_answer_length|answer text 생성 최대 길이 설정|30|
|eval_retrieval|원하는 retrieval 선택|sparse|
|num_clusters|faiss 사용 시, cluster 갯수 지정|64|
|top_k_retrieval|retrieve 시, 유사도 top k만큼의 passage 정의|50|
|score_ratio|score ratio 정의|0|
|train_retrieval|sparse/dense embedding을 train에 사용 유무 결정|False|
|data_selected|context or answers or question 중, 추가할 Unknown token 설정|""|
|rtt_dataset_name|RTT data path 설정|None|
|preprocessing_pattern|원하는 전처리 선택|None|
|add_special_tokens_flag|special token 추가|False|
|add_special_tokens_query_flag|Question type에 관한 speical token 추가|False|
|retrieve_pickle|pickle file 넣기|''|
|another_scheduler_flag|다른 scheduler 사용|False|
|num_cycles|cosine schedule with warmup cycle 설정|1|


### LoggingArguments

|     argument          | description                                                                                   | default        |
| :---------------:     | :-------------------------------------------------------------------------------------------- | :------------- |
|     wandb_name        | wandb에 기록될 모델의 이름                                                                            | model/roberta  |
| dotenv_path    | wandb key값을 등록하는 파일의 경로  | ./wandb.env |
|project_name|wandb에 기록될 project name|False |


## Running Command
### Train
```
$ python train.py --output_dir ./models --do_train --preprocessing_pattern 0 --add_special_tokens_query_flag True
```

### Reader evaluation

```
$ python train.py --output_dir ./outputs --do_eval --model_name_or_path ./models --preprocessing_pattern 0 --add_special_tokens_query_flag True
```

### ODQA evaluation

```
$ python inference.py --output_dir ./outputs --do_eval --model_name_or_path ./models --preprocessing_pattern 0 --add_special_tokens_query_flag True --top_k_retrieval 100 --score_ratio 0.85
```

### Inference prediction

```
$ python inference.py --output_dir ./outputs --do_predict --model_name_or_path ./models --preprocessing_pattern 0 --add_special_tokens_query_flag True  --dataset_name ../data/test_dataset/ --top_k_retrieval 100 --score_ratio 0.85
```

### Soft-voting Ensemble

단일 모델의 결과 nbest_predictions.json 파일들에서 probability 기반 soft-voting 하여 최종 ensemble 결과 json을 생성합니다.

### Hard-voting Ensemble

단일 모델의 결과 predictions.json 파일들에서 빈도 기반 hard-voting 하여 최종 ensemble 결과 json을 생성합니다.

> utils/nbest_ensemble.ipynb


## Reference
1. Dense Passage Retrieval for Open-Domain Question Answering 
    > https://arxiv.org/abs/2004.04906
2. Passage Re-Ranking With BERT 
    > https://arxiv.org/pdf/1901.04085.pdf
3. Latent Retrieval for Weekly Supervised Open Domain Question Answering 
    > https://arxiv.org/pdf/1906.00300.pdf
4. Cheap and Good? : Simple and Effective Data Augmentation for Low Source Machine Reading 
    > https://arxiv.org/abs/2106.04134
5. How NLP Can Improve Question Answering 
    > https://core.ac.uk/download/pdf/31832115.pdf