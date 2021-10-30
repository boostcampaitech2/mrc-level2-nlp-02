import os
import pickle
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from transformers import (
    AdamW,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from rt_arguments import (
    EncoderModelArguments,
    RtDataTrainingArguments,
)
from rt_model import klueRobertaEncoder
from rt_dataset import RtTrainDataset

def save_encoder(model_name, p_encoder, q_encoder, training_args, data_args):
    # 파일 이름 model_epoch_batch_topk_acs_pp(preprocessing pattern)
    file_name = f"{model_name.split('/')[-1]}_ep{int(training_args.num_train_epochs)}_bs{training_args.per_device_train_batch_size}_topk{data_args.top_k_retrieval}_acs{training_args.gradient_accumulation_steps}_pp{data_args.preprocessing_pattern}"
    full_path = os.path.join(training_args.output_dir, file_name)

    os.makedirs(full_path, exist_ok=True)

    p_encoder_path = os.path.join(full_path, "passage.pt")
    q_encoder_path = os.path.join(full_path, "query.pt")

    torch.save(p_encoder.state_dict(), p_encoder_path)
    torch.save(q_encoder.state_dict(), q_encoder_path)
        

def train(args, p_encoder, q_encoder, train_dataloader, valid_dataloader, top_k):
    print("training by bm25 negative sampling data !!!")
    batch_size = args.per_device_train_batch_size
    valid_batch_size = args.per_device_eval_batch_size

    accumulation_step = args.gradient_accumulation_steps
    print(f"accumulation_step : {accumulation_step}")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    t_total = (len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    scaler = GradScaler()

    p_encoder.to(args.device)
    q_encoder.to(args.device)
    p_encoder.zero_grad()
    q_encoder.zero_grad()
    optimizer.zero_grad()

    torch.cuda.empty_cache()

    epochs = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

    for _ in epochs:
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for idx, batch in enumerate(tepoch):
                p_encoder.train()
                q_encoder.train()

                if len(batch) == 5:
                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * top_k, -1).to(args.device),  # (batch_size * tok_k, emb_dim)
                        "attention_mask": batch[1].view(batch_size * top_k, -1).to(args.device),  # (batch_size, tok_k, emb_dim)
                    }
                    q_inputs = {
                        "input_ids": batch[2].view(batch_size, -1).to(args.device),  # (batch_size, emb_dim)
                        "attention_mask": batch[3].view(batch_size, -1).to(args.device),  # (batch_size, emb_dim)
                    }
                    targets = batch[4].long().to(args.device)
                else:
                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * top_k, -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * top_k, -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * top_k, -1).to(args.device),
                    }
                    q_inputs = {
                        "input_ids": batch[3].view(batch_size, -1).to(args.device),
                        "attention_mask": batch[4].view(batch_size, -1).to(args.device),
                        "token_type_ids": batch[5].view(batch_size, -1).to(args.device),
                    }
                    targets = batch[6].long().to(args.device)

                with autocast():
                    
                    p_outputs = p_encoder(**p_inputs)
                    q_outputs = q_encoder(**q_inputs)

                    p_outputs = p_outputs.view(batch_size, top_k, -1)  # (batch_size, tok_k, emb_dim)
                    q_outputs = q_outputs.view(batch_size, 1, -1)  # (batch_size, 1, emb_dim)

                    sim_scores = torch.bmm(q_outputs, p_outputs.transpose(1, 2)).squeeze()  # (batchsize, top_k)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets) / accumulation_step
                    tepoch.set_postfix(loss=f"{loss.item():.4f}")

                scaler.scale(loss).backward()

                if ((idx + 1) % accumulation_step == 0) or ((idx + 1) == len(train_dataloader)):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    optimizer.zero_grad()
                    p_encoder.zero_grad()
                    q_encoder.zero_grad()

                    del p_inputs
                    del q_inputs
                    
                if ((idx + 1) % args.eval_steps == 0) or ((idx + 1) == len(train_dataloader)):
                    total_cnt = len(valid_dataloader.dataset)
                    valid_loss = 0
                    valid_acc = 0
                    for valid_batch in tqdm(valid_dataloader):
                        with torch.no_grad():
                            p_encoder.eval()
                            q_encoder.eval()

                            if len(batch) == 5:
                                p_inputs = {
                                    "input_ids": valid_batch[0].view(valid_batch_size * top_k, -1).to(args.device),  # (batch_size * tok_k, emb_dim)
                                    "attention_mask": valid_batch[1].view(valid_batch_size * top_k, -1).to(args.device),  # (batch_size, tok_k, emb_dim)
                                }
                                q_inputs = {
                                    "input_ids": valid_batch[2].view(valid_batch_size, -1).to(args.device),  # (batch_size, emb_dim)
                                    "attention_mask": valid_batch[3].view(valid_batch_size, -1).to(args.device),  # (batch_size, emb_dim)
                                }
                                targets = valid_batch[4].long().to('cpu')
                            else:
                                p_inputs = {
                                    "input_ids": valid_batch[0].view(valid_batch_size * top_k, -1).to(args.device),
                                    "attention_mask":valid_batch[1].view(valid_batch_size * top_k, -1).to(args.device),
                                    "token_type_ids": valid_batch[2].view(valid_batch_size * top_k, -1).to(args.device),
                                }
                                q_inputs = {
                                    "input_ids": valid_batch[3].view(valid_batch_size, -1).to(args.device),
                                    "attention_mask": valid_batch[4].view(valid_batch_size, -1).to(args.device),
                                    "token_type_ids": valid_batch[5].view(valid_batch_size, -1).to(args.device),
                                }
                                targets = valid_batch[6].long().to('cpu')
                            p_outputs = p_encoder(**p_inputs).to('cpu')
                            q_outputs = q_encoder(**q_inputs).to('cpu')

                            p_outputs = p_outputs.view(valid_batch_size, top_k, -1)  # (batch_size, tok_k, emb_dim)
                            q_outputs = q_outputs.view(valid_batch_size, 1, -1)  # (batch_size, 1, emb_dim)

                            sim_scores = torch.bmm(q_outputs, p_outputs.transpose(1, 2)).squeeze()  # (batchsize, top_k)
                            sim_scores = sim_scores.view(valid_batch_size, -1)

                            prediction = torch.argmax(sim_scores, dim=-1)

                            valid_acc += (prediction == targets).sum().item()
                            valid_loss += F.cross_entropy(sim_scores, targets).item()

                    print(f"valid accuracy : {valid_acc/total_cnt:.2%}")
                    print(f"valid loss : {valid_loss/len(valid_dataloader):.3f}")
                             

    return p_encoder, q_encoder

def main(model_args, data_args, training_args):
    print(f"Train !!!")
    print(f"model is from {encoder_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    set_seed(training_args.seed)

    if data_args.preprocessing_pattern == None:
        data_args.preprocessing_pattern = 0

    train_custom_pickle = os.path.join(
        data_args.train_pickle_save_dir, f"bm25_top{data_args.top_k_retrieval}_pp{data_args.preprocessing_pattern}.pickle"
    )
    valid_custom_pickle = os.path.join(
        data_args.valid_pickle_save_dir, f"bm25_top{data_args.top_k_retrieval}_pp{data_args.preprocessing_pattern}.pickle"
    )

    print(f"train custom data is from {train_custom_pickle}")
    print(f"valid custom data is from {valid_custom_pickle}")

    with open(train_custom_pickle, "rb") as f:
        train_dataset = pickle.load(f)
    with open(valid_custom_pickle, "rb") as f:
        valid_dataset = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )
    p_encoder = klueRobertaEncoder(model_args.model_name_or_path)
    q_encoder = klueRobertaEncoder(model_args.model_name_or_path)

    rt_train_dataset = RtTrainDataset(
        train_dataset, tokenizer, model_name=model_args.model_name_or_path
    )
    rt_valid_dataset = RtTrainDataset(
        valid_dataset, tokenizer, model_name=model_args.model_name_or_path
    )

    rt_train_loader = DataLoader(
        rt_train_dataset,
        shuffle=True,
        batch_size=training_args.per_device_train_batch_size,
    )
    rt_valid_loader = DataLoader(
        rt_valid_dataset,
        batch_size=training_args.per_device_eval_batch_size,
    )
    print(f"length of train dataset : {len(rt_train_dataset)}")
    print(f"length of train dataloader : {len(rt_train_loader)}")
    print(f"length of valid dataset : {len(rt_valid_dataset)}")
    print(f"length of valid dataloader : {len(rt_valid_loader)}")

    p_encoder, q_encoder = train(
        training_args, p_encoder, q_encoder, rt_train_loader, rt_valid_loader, data_args.top_k_retrieval
    )

    save_encoder(encoder_args.model_name_or_path, p_encoder, q_encoder, training_args, data_args)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (EncoderModelArguments, RtDataTrainingArguments, TrainingArguments)
    )

    encoder_args, rt_data_args, training_args = parser.parse_args_into_dataclasses()

    main(encoder_args, rt_data_args, training_args)

