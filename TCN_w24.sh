# TCN
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Aggregate --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Aggregate --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance1 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance1 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance2 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance2 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance3 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance3 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance4 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance4 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance5 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance5 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance6 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance6 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance7 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance7 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance8 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance8 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance9 --window_size 24 --model TCN --layer 4 --patience 10 --save tuning_w24/TCN_Appliance9 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2

# DNN
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Aggregate --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Aggregate --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance1 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance1 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance2 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance2 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance3 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance3 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance4 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance4 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance5 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance5 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance6 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance6 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance7 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance7 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance8 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance8 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance9 --window_size 24 --model ANN --patience 10 --save tuning_w24/ANN_Appliance9 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2

# LSTM
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Aggregate --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Aggregate --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance1 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance1 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance2 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance2 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance3 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance3 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance4 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance4 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance5 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance5 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance6 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance6 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance7 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance7 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance8 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance8 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance9 --window_size 24 --model LSTM --patience 10 --save tuning_w24/LSTM_Appliance9 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 2
