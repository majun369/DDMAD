python main.py --batch_size 256 --win_size 100 --d_model 100 --kernel_size 20 --fp 2 --gpu 0 --lr 2e-4 --num_epochs 3 --warm_epochs 1 --input_c 25 --output_c 25 --e_layers 1 --k 3 --dataset SMAP --mode train --data_path SMAP --model_save_path checkpoints/SMAP --anormly_ratio 0.85
python main.py --batch_size 256 --win_size 100 --d_model 100 --kernel_size 20 --gpu 0 --input_c 25 --output_c 25 --e_layers 1 --k 3 --dataset SMAP --mode test --data_path SMAP --model_save_path checkpoints/SMAP --anormly_ratio 0.85


