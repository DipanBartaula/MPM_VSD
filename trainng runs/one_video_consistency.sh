#!/bin/bash
# Single Video Render with Frame Consistency Regularization
python ../train_sds_physics.py \
    --trained_model_path ../pretrained_models/output/tracking/a1_s1_460_200 \
    --model_path ../pretrained_models/model/a1_s1 \
    --dataset_dir ../data \
    --output_dir ../output \
    --actor 1 --sequence 1 \
    --train_frame_start_num 460 32 \
    --verts_start_idx 460 \
    --wan_ckpt_dir ../wan_5b_model \
    --sds_cfg ../bridge_sds/configs/sds_test.yaml \
    --iterations 2000 \
    --save_name run_spsa_1cam_temporal_consistency_reg \
    --num_cams 1 \
    --use_consistency_reg \
    --consistency_weight 0.1 \
    --random_init_params
