from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EncoderModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large",
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
        metadata={"help": "customized tokenizer path if not the same as model_name"},
    )
    customized_tokenizer_flag: bool = field(
        default=False,
        metadata={"help": "Load customized roberta tokenizer"},
    )


@dataclass
class RtDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="/opt/ml/data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    data_path: Optional[str] = field(
        default="/opt/ml/data/",
        metadata={"help": "The path of wiki dir"},
    )
    context_path: Optional[str] = field(
        default="wikipedia_documents.json",
        metadata={"help": "The name of the wiki file"},
    )
    top_k_retrieval: int = field(
        default=1,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    save_dir: Optional[str] = field(
        default="/opt/ml/data/bm25",
        metadata={"help": "The path of bm25 csv"},
    )
    train_train_pickle_save_dir: Optional[str] = field(
        default="/opt/ml/data/train_pickle",
        metadata={"help": "The path of custom_dataset"},
    )
    valid_train_pickle_save_dir: Optional[str] = field(
        default="/opt/ml/data/valid_pickle",
        metadata={"help": "The path of custom_dataset"},
    )
    preprocessing_pattern: str = field(
        default=None,
        metadata={"help": "preprocessing(e.g. 123)"},
    )
    score_ratio: float = field(
        default=0,
        metadata={"help": "Define the score ratio."},
    )
