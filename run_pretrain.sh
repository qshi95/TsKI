CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tabfact.py \
  --model_name_or_path bert-base-multilingual-cased \
  --task_name tab_fact_pretrain \
  --train_file PATH_TO_TRAIN_DATA \
  --validation_file PATH_TO_VAL_DATA \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --output_dir PATH_TO_OUTPUT \
  --evaluation_strategy steps \
  --warmup_steps 10000 \
  --weight_decay 0.001 \
  --fp16 \
  --mode pretrain


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_all.py \
  --model_name_or_path google/tapas-base \
  --task_name tab_fact_pretrain \
  --train_file PATH_TO_TRAIN_DATA \
  --validation_file PATH_TO_VAL_DATA \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 20.0 \
  --output_dir PATH_TO_OUTPUT \
  --evaluation_strategy steps \
  --warmup_steps 10000 \
  --weight_decay 0.001 \
  --fp16 \
  --mode pretrain

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_all.py \
  --model_name_or_path microsoft/tapex-base \
  --task_name tab_fact_pretrain \
  --train_file PATH_TO_TRAIN_DATA \
  --validation_file PATH_TO_VAL_DATA \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 20.0 \
  --output_dir PATH_TO_OUTPUT \
  --evaluation_strategy steps \
  --warmup_steps 10000 \
  --weight_decay 0.001 \
  --fp16 \
  --mode pretrain