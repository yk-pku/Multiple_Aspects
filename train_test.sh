name='train_full'
model_save_dir='results_10%'
semi_ratio='0.1'
epoch='240'
CUDA_VISIBLE_DEVICES=3,4 \
OMP_NUM_THREADS=4 \
torchrun --master_port 9840 --nproc_per_node=2 train.py \
  --name ${name} \
  --model_save_dir ./${model_save_dir}/ \
  --train_txt ./dataset_split/train_AtriaSeg.txt \
  --val_txt ./dataset_split/test_AtriaSeg.txt \
  --label_factor_semi ${semi_ratio} \
  --epochs ${epoch} \
  --with_dice 0 \
  --pseduo_threshold 0.9 \
  --noise_branch \
&& \
python test.py\
    --model_dir ./${model_save_dir}/${name}/\
    --test_txt ./dataset_split/test_AtriaSeg.txt \
    --num_epoch _${epoch}
