# TCN
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Aggregate --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Aggregate --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance1 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance1 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance2 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance2 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance3 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance3 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance4 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance4 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance5 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance5 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance6 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance6 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance7 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance7 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance8 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance8 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance9 --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Appliance9 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9

# DNN
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Aggregate --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Aggregate --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance1 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance1 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance2 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance2 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance3 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance3 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance4 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance4 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance5 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance5 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance6 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance6 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance7 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance7 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance8 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance8 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance9 --window_size 6 --model ANN --patience 10 --save tuning_test/ANN_Appliance9 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9

# LSTM
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Aggregate --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Aggregate --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance1 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance1 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance2 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance2 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance3 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance3 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance4 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance4 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance5 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance5 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance6 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance6 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance7 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance7 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance8 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance8 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Appliance9 --window_size 6 --model LSTM --patience 10 --save tuning_test/LSTM_Appliance9 --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
