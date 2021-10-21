# ## WLPR1
# python train.py --output_dir ./models/KRL_WLPR_ly4 --do_train --model_name_or_path klue/roberta-large --reader_custom_model WeightedLayerPoolingRobertaQA --wandb_tag WLPR_ly4
# python train.py --output_dir ./models/KRL_WLPR_ly4 --do_eval --model_name_or_path ./models/KRL_WLPR_ly4 --reader_custom_model WeightedLayerPoolingRobertaQA --wandb_tag WLPR_ly4
# python inference.py --output_dir ./models/KRL_WLPR_ly4 --do_eval --dataset_name ../data/train_dataset/ --model_name_or_path ./models/KRL_WLPR_ly4 --reader_custom_model WeightedLayerPoolingRobertaQA --wandb_tag WLPR_ly4
# python inference.py --output_dir ./outputs/KRL_WLPR_ly4 --do_predict --dataset_name ../data/test_dataset/ --model_name_or_path ./models/KRL_WLPR_ly4 --reader_custom_model WeightedLayerPoolingRobertaQA


# ## WLPR_ly2
# python train.py --output_dir ./models/KRL_WLPR_ly2 --do_train --model_name_or_path klue/roberta-large --reader_custom_model WeightedLayerPoolingRobertaQA --pooled_lalyer_start -2 --wandb_tag WLPR_ly2
# python train.py --output_dir ./models/KRL_WLPR_ly2 --do_eval --model_name_or_path ./models/KRL_WLPR_ly2 --reader_custom_model WeightedLayerPoolingRobertaQA --pooled_lalyer_start -2 --wandb_tag WLPR_ly2
# python inference.py --output_dir ./models/KRL_WLPR_ly2 --do_eval --dataset_name ../data/train_dataset/ --model_name_or_path ./models/KRL_WLPR_ly2 --reader_custom_model WeightedLayerPoolingRobertaQA --pooled_lalyer_start -2 --wandb_tag WLPR_ly2
# python inference.py --output_dir ./outputs/KRL_WLPR_ly2 --do_predict --dataset_name ../data/test_dataset/ --model_name_or_path ./models/KRL_WLPR_ly2 --reader_custom_model WeightedLayerPoolingRobertaQA --pooled_lalyer_start -2

# ## WLPR_ly8
# python train.py --output_dir ./models/KRL_WLPR_ly8 --do_train --model_name_or_path klue/roberta-large --reader_custom_model WeightedLayerPoolingRobertaQA --pooled_lalyer_start -8 --wandb_tag WLPR_ly8
# python train.py --output_dir ./models/KRL_WLPR_ly8 --do_eval --model_name_or_path ./models/KRL_WLPR_ly8 --reader_custom_model WeightedLayerPoolingRobertaQA --pooled_lalyer_start -8 --wandb_tag WLPR_ly8
# python inference.py --output_dir ./models/KRL_WLPR_ly8 --do_eval --dataset_name ../data/train_dataset/ --model_name_or_path ./models/KRL_WLPR_ly8 --reader_custom_model WeightedLayerPoolingRobertaQA --pooled_lalyer_start -8 --wandb_tag WLPR_ly8
# python inference.py --output_dir ./outputs/KRL_WLPR_ly8 --do_predict --dataset_name ../data/test_dataset/ --model_name_or_path ./models/KRL_WLPR_ly8 --reader_custom_model WeightedLayerPoolingRobertaQA --pooled_lalyer_start -8

# XLM-robeta
# python train.py --output_dir ./models/XRL --do_train --model_name_or_path xlm-roberta-large
# python train.py --output_dir ./models/XRL --do_eval --model_name_or_path ./models/XRL
## infer 결과가 이상한데 아마 tokenizer 때문에???
# python inference.py --output_dir ./models/XRL --do_eval --dataset_name ../data/train_dataset/ --model_name_or_path ./models/XRL
# python inference.py --output_dir ./outputs/XRL --do_predict --dataset_name ../data/test_dataset/ --model_name_or_path ./models/XRL

# python train.py --output_dir ./models/XRL_WLPR_ly4 --do_train --model_name_or_path xlm-roberta-large --wandb_tag WLPR_ly4
# python train.py --output_dir ./models/XRL_WLPR_ly4 --do_eval --model_name_or_path ./models/XRL_WLPR_ly4 --wandb_tag WLPR_ly4
# python inference.py --output_dir ./models/XRL_WLPR_ly4 --do_eval --dataset_name ../data/train_dataset/ --model_name_or_path ./models/XRL_WLPR_ly4 --wandb_tag WLPR_ly4
# python inference.py --output_dir ./outputs/XRL_WLPR_ly4 --do_predict --dataset_name ../data/test_dataset/ --model_name_or_path ./models/XRL_WLPR_ly4


## LSTM
# echo 'train'
# python train.py --output_dir ./models/KRL_LSTM_test --do_train --model_name_or_path klue/roberta-small --reader_custom_model RobertaQALSTM --wandb_tag ly3
# echo 'train eval'
# python train.py --output_dir ./models/KRL_LSTM_test --do_eval --model_name_or_path ./models/KRL_LSTM_test --reader_custom_model RobertaQALSTM --wandb_tag ly3
# echo 'infer eval'
# python inference.py --output_dir ./models/KRL_LSTM_test --do_eval --dataset_name ../data/train_dataset/ --model_name_or_path ./models/KRL_LSTM_test --reader_custom_model RobertaQALSTM --wandb_tag ly3

## Koelectra
# python train.py --output_dir ./models/koelectra_base --do_train --model_name_or_path monologg/koelectra-base-v3-discriminator
# python train.py --output_dir ./models/koelectra_base --do_eval --model_name_or_path ./models/koelectra_base
# python inference.py --output_dir ./models/koelectra_base --do_eval --dataset_name ../data/train_dataset/ --model_name_or_path ./models/koelectra_base

## Roberta
python train.py --output_dir ./models/klue_roberta-large --do_train --model_name_or_path klue/roberta-large
python train.py --output_dir ./models/klue_roberta-large --do_eval --model_name_or_path ./models/klue_roberta-large
python inference.py --output_dir ./models/klue_roberta-large --do_eval --dataset_name ../data/train_dataset/ --model_name_or_path ./models/klue_roberta-large