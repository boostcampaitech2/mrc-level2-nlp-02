from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    rt_model_name: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default="klue/roberta-large",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "customized tokenizer path if not the same as model_name"
        },
    )
    customized_tokenizer_flag: bool = field(
        default=False,
        metadata={"help": "Load customized roberta tokenizer"},
    )
    k_fold : int = field(
        default=5,
        metadata={"help": "K for K-fold validation"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="/opt/ml/data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=2,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True, #True
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: str = field(
        default="sparse",
        metadata={
            "help": "Choose which passage retrieval to be used.[sparse, elastic_sparse]."
        },
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=50,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    score_ratio: float = field(
        default=0,
        metadata={
            "help": "Define the score ratio."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    train_retrieval: bool = field(
        default=False,
        metadata={"help": "Whether to train sparse/dense embedding (prepare for retrieval)."},
    )
    data_selected: str = field(
        default="",
        metadata={"help": "data to find added tokens, context/answers/question with '_' e.g.) context_answers"},
    )
    rtt_dataset_name:str = field(
        default=None,
        metadata={"help" : "input rtt data name with path"},
    )
    preprocessing_pattern:str = field(
        default=None,
        metadata={"help" : "preprocessing(e.g. 123)"},
    )
    add_special_tokens_flag:bool = field(
        default=False,
        metadata={"help": "add special tokens"},
    )
    pretrain_span_augmentation : bool = field(
        default=False,
        metadata={"help": "pretrain data using span augmentation"},
    )
    num_neg: int = field(
        default=0, metadata={"help": "Define how many negative sampling dataset"},
    )
    reconfig: bool = field(
        default=False,
        metadata={"help": "Elastic search re-config flag"},
    )
    re_rank: bool = field(
        default=False,
        metadata={"help": "re-rank top-k passage"},
    )
    re_rank_top_k : int = field(
        default=10, metadata={"help": "Define how many re-rank passage"},
    )
    add_special_tokens_query_flag:bool = field(
        default=False,
        metadata={"help": "add special tokens about question type"},
    )

@dataclass
class LoggingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    wandb_name: Optional[str] = field(
        default="model/roberta",
        metadata={"help": "The name of the dataset to use."},
    )

    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )

    project_name: Optional[str] = field(
        default="mrc_project_1",
        metadata={"help": "The name of the dataset to use."},
    )
