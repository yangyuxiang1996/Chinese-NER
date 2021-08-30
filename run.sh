CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/chinese_roberta_wwm_ext
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
# TASK_NAME="cluener"
# TASK_NAME="jdner"
TASK_NAME="wbner"
MODEL_TYPE="bilstm"

python run.py \
  --model_type=$MODEL_TYPE \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_crf \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=2 \
  --logging_steps=10 \
  --save_steps=10 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42 \
  --embedding_size=1024 \
  --hidden_size=512