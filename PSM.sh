python main.py --batch_size 256 --d_model 16 --fp 1.5 --win_size 100 --kernel_size 25 --gpu 0 --lr 2e-4 --input_c 25 --output_c 25 --e_layers 1 --num_epochs 4 --dataset PSM --mode train --data_path PSM --model_save_path checkpoints/PSM --anormly_ratio 1.0
python main.py --batch_size 256 --d_model 16 --win_size 100 --kernel_size 25 --gpu 0 --input_c 25 --output_c 25 --e_layers 1 --dataset PSM --mode test --data_path PSM --model_save_path checkpoints/PSM --anormly_ratio 1.0

