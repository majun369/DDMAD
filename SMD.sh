python main.py --batch_size 256 --win_size 100 --d_model 100 --fp 2 --kernel_size 20 --gpu 0 --lr 2e-4 --input_c 38 --output_c 38 --warm_epochs 1 --num_epochs 3 --e_layers 1 --k 3 --dataset SMD --mode train --data_path SMD --model_save_path checkpoints/SMD --anormly_ratio 0.5
python main.py --batch_size 256 --win_size 100 --d_model 100 --kernel_size 20 --gpu 0 --input_c 38 --output_c 38 --e_layers 1 --k 3 --dataset SMD --mode test --data_path SMD --model_save_path checkpoints/SMD --anormly_ratio 0.5





