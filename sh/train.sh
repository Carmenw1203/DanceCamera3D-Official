accelerate launch scripts/train.py \
--project runs/train_SepCFG_CentricCam \
--data_path ../../data/wangzx_DCM/data/amc_data_split_by_categories \
--render_dir renders/DanceCamera3D_SepCFG_CentricCam \
--wandb_pj_name DanceCamera3D_SepCFG_CentricCam \
--batch_size 128  --epochs 3000 \
--feature_type jukebox \
--learning_rate 0.0002 \
--save_interval 500 \
--camera_format polar \
--condition_separation_CFG True \
--w_loss 2 \
--w_v_loss 5 \
--w_a_loss 5 \
--w_in_ba_loss 0.0015 \
--w_out_ba_loss 0



