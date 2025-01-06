python main.py --batch_size 256 --win_size 100 --d_model 320 --kernel_size 25 --lr 2e-4 --gpu 0 --input_c 51 --output_c 51 --num_epochs 4 --e_layers 3 --dataset SWAT --mode train --data_path SWAT --model_save_path checkpoints/SWAT --anormly_ratio 1.0
python main.py --batch_size 256 --win_size 100 --d_model 320 --kernel_size 25 --gpu 0 --input_c 51 --output_c 51 --e_layers 3 --k 3 --dataset SWAT --mode test --data_path SWAT --model_save_path checkpoints/SWAT --anormly_ratio 1.0

