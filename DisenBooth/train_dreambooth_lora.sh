export MODEL_NAME="/mnt/workspace/workgroup_share/lhn/models/stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="./dog7"
export CLASS_DIR="./output/dreambooth_lora/class_imgs"
export OUTPUT_DIR=".output/dreambooth_lora"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=200 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=5 \
  --max_train_steps=2000 \
  --validation_prompt="A dog</w> dog in a bucket" \
  --validation_epochs=200 \
  --seed="0" \

#   --push_to_hub