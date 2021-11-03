import pickle

from torch.utils.data import Dataset, DataLoader

from typing import Dict

from transformers import AutoTokenizer

from rt_model import klueRobertaEncoder


class RtTrainDataset(Dataset):
    def __init__(
        self,
        dataset: Dict,
        tokenizer,
        model_name="klue/roberta",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_name = model_name

    def __len__(self):
        return len(self.dataset["question"])

    def __getitem__(self, idx):
        question = self.dataset["question"][idx]
        context = self.dataset["top_k_passage"][idx]
        target = self.dataset["target"][idx]

        p_seqs = self.tokenizer(
            context,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False if "roberta" in self.model_name else True,
        )  # (top_k, 512)

        q_seqs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False if "roberta" in self.model_name else True,
        )  # (1, 512)

        if "roberta" in self.model_name:
            return (
                p_seqs["input_ids"],
                p_seqs["attention_mask"],
                q_seqs["input_ids"],
                q_seqs["attention_mask"],
                target,
            )
        else:
            return (
                p_seqs["input_ids"],
                p_seqs["attention_mask"],
                p_seqs["token_type_ids"],
                q_seqs["input_ids"],
                q_seqs["attention_mask"],
                q_seqs["token_type_ids"],
                target,
            )


# if __name__ == "__main__":
#     model_name = "klue/roberta-large"

#     with open("/opt/ml/data/train_pickle/bm25_5.pickle", "rb") as f:
#         dataset = pickle.load(f)

#     print(
#         dataset["question"][0],
#         len(dataset["top_k_passage"]),
#     )

#     tokinzer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     p_encoder = klueRobertaEncoder(model_name)
#     q_encoder = klueRobertaEncoder(model_name)

#     rt_train_dataset = RtTrainDataset(dataset, tokinzer, model_name=model_name)
#     print(len(rt_train_dataset[0]))

#     rt_train_loader = DataLoader(rt_train_dataset, shuffle=True, batch_size=4)
#     print(len(rt_train_loader))

#     for batch in rt_train_loader:
#         breakpoint()
