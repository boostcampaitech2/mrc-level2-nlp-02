from transformers import set_seed, TrainingArguments

from arguments import EncoderModelArguments, DataTrainingArguments, HfArgumentParser


def main(args):
    pass


if __name__ == "__main__":
    parser = HfArgumentParser(
        (EncoderModelArguments, DataTrainingArguments, TrainingArguments)
    )

    encoder_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"model is from {encoder_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")
