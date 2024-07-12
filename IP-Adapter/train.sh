# accelerate launch --num_processes 1 --mixed_precision "fp16" \
python3 -m pdb tutorial_train.py \
  --pretrained_model_name_or_path="/mnt/workspace/workgroup/xdj/models/AIGC/sd_model/model_path/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0/" \
  --image_encoder_path="/mnt/workspace/workgroup/xdj/models/AIGC/sd_model/model_path/models/Ip-adpter/image_encoder" \
  --data_json_file="/mnt/workspace/workgroup_share/lhn/IP-Adapter/data.json" \
  --data_root_path="/mnt/workspace/workgroup_share/lhn/IP-Adapter/assets/demo" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="output" \
  --save_steps=10000 \
  --train_batch_size 1


  #sd1.5:/mnt/workspace/workgroup/xdj/models/AIGC/sd_model/model_path/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0
