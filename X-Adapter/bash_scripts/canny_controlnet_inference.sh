python inference.py --plugin_type "controlnet" \
--prompt "A cute cat, high quality, extremely detailed" \
--condition_type "canny" \
--input_image_path "./assets/CuteCat.jpeg" \
--controlnet_condition_scale_list 1.5 \
--adapter_guidance_start_list 0.8 \
--adapter_condition_scale_list 1.0 \
--height 1024 \
--width 1024 \
--height_sd1_5 512 \
--width_sd1_5 512 \
--controlnet_canny_path /mnt/workspace/workgroup/xdj/models/AIGC/sd_model/model_path/models/control_v11p_sd15_canny
# #1.75 2.0 \

# #1.20 \