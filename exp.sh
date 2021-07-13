python eval/gen_video.py -n fluid_shake_more_epcs --gpu_id 2 --split test -P "0 1 2 3" -D '/home/htxue/src/water_pour_small/'  \
                         --checkpoints_path 'checkpoints/' --dataset_format 'fluid_shake' -S 1


python eval/gen_video.py -n SRN_CAR --gpu_id 2 --split test -P "0 1 2 3" -D '/home/htxue/src/srn_car/cars'  \
                         --checkpoints_path 'checkpoints/' --dataset_format 'srn' -S 1

python train/train.py -n fluid_shake_whitebkg --dataset_format 'fluid_shake'  -D '/home/htxue/src/water_pour_small/'  --gpu_id=2 --resume
python train/train.py -n fluid_shake_whitebkg --dataset_format 'fluid_shake'  -D '/home/htxue/src/water_pour_small/'  --gpu_id=2 --resume

python train/train.py -n SRN_CAR --dataset_format 'srn'  -D '/home/htxue/srn_car/cars'  --gpu_id=2

python eval/gen_video.py -n fluid_shake_more_epcs --gpu_id 1 --split test -P "0 1 2 3 4"  \
       -D '/home/htxue/src/water_pour_small/'  --checkpoints_path 'checkpoints/' --dataset_format 'fluid_shake' -S 1

# new
python train/train.py -n fluid_shake_0701_4_views --dataset_format 'fluid_shake'  -D '/home/htxue/data_Pour/'  --gpu_id=2




# ---------------------- 0707 ----------------------------- #
# train fluid_shake
python train/train.py -n fluid_shake_whitebkg \
       --visual_path visuals_0707 \
       --checkpoints_path checkpoints_0707 \
       --dataset_format fluid_shake  \
       -D /home/htxue/datasets/water_shake_small/  \
       --gpu_id=2
# train fluid_pour
python train/train.py -n fluid_pour_whitebkg \
       --visual_path visuals_0707 \
       --checkpoints_path checkpoints_0707 \
       --dataset_format fluid_pour  \
       -D /home/htxue/datasets/water_pour_small/  \
       --gpu_id=2