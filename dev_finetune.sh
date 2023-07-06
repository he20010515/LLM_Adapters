method=$1
python finetune.py \
  --base_model 'decapoda-research/llama-7b-hf' \
  --data_path 'math_data.json' \
  --output_dir ./trained_models/llama-${method} \
  --batch_size 4 \
  --micro_batch_size 4 \
  --num_epochs 6 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name ${method} \
  # --use_global_kv_adapter \
  # --codebook_nums $codebook_nums \
  # --num_memories $num_memories \
