python main.py --d_model 16 --fp 1.5 --gpu 0 --input_c 55 --output_c 55 --e_layers 3 --dataset MSL --mode train --data_path MSL --model_save_path checkpoints/MSL --anormly_ratio 1.0
python main.py --d_model 16 --gpu 0 --input_c 55 --output_c 55 --e_layers 3 --dataset MSL --mode test --data_path MSL --model_save_path checkpoints/MSL --anormly_ratio 1.0

