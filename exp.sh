python eval/gen_video.py -n sn64 --gpu_id 2 --split test -P '2' -D '/home/htxue/src/water_pour_small/'  \
                         --checkpoints_path 'checkpoints/' --dataset_format 'fluid_shake' -S 1




python train/train.py -n fluid_shake --dataset_format 'fluid_shake'  -D '/home/htxue/src/water_pour_small/'  --gpu_id=2 --resume




python train/train.py -n fluid_shake --dataset_format 'fluid_shake'  -D '/home/htxue/src/water_pour_small/'  --gpu_id=2 --resume