from numpy.random import seed 
seed(42)  ## 设置伪随机序列
from tensorflow.random import set_seed
set_seed(42)

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils.vis_utils import plot_model
from model import build_model
from tools import rmse,mae,mse,mape,Standardscale,Maxminscale,Nonescale
from vmd import VMD
        
parser = argparse.ArgumentParser(description="Electrical load curve prediction")
parser.add_argument("--data", type=str, default="REFIT_processed\RAW_House1_12H_processed.csv", 
                    help="a csv file path which include some appliance load data ")
parser.add_argument("--target", type=str, default='Aggregate',
                    help="predict target cure including Aggregate or Appliancex")
parser.add_argument("--test_rate", type=float, default=0.2, 
                    help="test set rate")
parser.add_argument("--window_size", type=int, default=6, 
                    help="Historical reference length")
parser.add_argument("--scale", type=str, default="standard", 
                    help="data scale for model inputs")                
parser.add_argument("--save", type=str, default=os.getcwd()+"/tuning_univariate/",
                    help="training result folder")
parser.add_argument("--model", type=str, default="CA",
                    help="Model name")
parser.add_argument("--layer", type=int, default=2,
                    help="TCN layers")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--vmd", action='store_true',
                    help="sue vmd to extract feature")
parser.add_argument("--patience",type=int, default=10,
                    help="early stop")
parser.add_argument("--epoch",type=int, default=1000,
                    help="training epoch")
parser.add_argument("--batchsize",type=int, default=128,
                    help="batchsize")
parser.add_argument("--gpu",type=int, default=0,
                    help="gpu device id")
                    
args = parser.parse_args()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # 设置 TensorFlow 只使用第一块 GPU
    try:
        tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 异常处理
        print(e)

if not os.path.exists(args.save):
    os.makedirs(args.save)
log_dic = {"data":{}, "model":{}, "traing":{}, "testing":{}}
plot = False

####################################################################################
#   Load Data
####################################################################################
def make_data(file_path, test_rate, window_size,scale, target): 

    df = pd.read_csv(file_path, parse_dates=["Time"], index_col ="Time")
    df = df[target]

    # 切分数据集
    test_num = int(len(df) * test_rate)
    train_data = df[:-test_num].copy().values
    test_data  = df[-test_num:].copy().values
    print("dataset shape:",df.shape)
    print("train_data shape:",train_data.shape)
    print("test_data shape:",test_data.shape)

    scale.fit(train_data)
    train_data = scale.transform(train_data)

    # 测试集数据归一化
    test_data = scale.transform(test_data)

    # 窗口滑动制作数据样本
    train_X = []
    train_y = []
    train_num = train_data.shape[0]
    for k in range(train_num-window_size):
        train_X.append(train_data[k:k+window_size]) #x: [k:k+window_size)
        train_y.append(train_data[k+window_size]) #y: k+windowsize 

    test_X = []
    test_y = []
    test_num = test_data.shape[0]
    for k in range(test_num-window_size):
        test_X.append(test_data[k:k+window_size]) #x: [k: k+window_size)
        test_y.append(test_data[k+window_size]) #y: k+windowsize 

    train_X, train_y = np.array(train_X), np.reshape(train_y,(len(train_y),1))
    test_X, test_y = np.array(test_X), np.reshape(test_y,(len(test_y),1)) 

    print("train_x shape: ", train_X.shape)
    print("train_y shape: ", train_y.shape)
    print("test_x shape: ", test_X.shape)
    print("test_y shape: ", test_y.shape)
    
    return train_X,train_y,test_X,test_y,scale

scale = Standardscale() 
train_X,train_y,test_X,test_y,scale = make_data(args.data, args.test_rate, args.window_size, scale, args.target)
log_dic["data"]["file"] = args.data
log_dic["data"]["target"] = args.target
log_dic["data"]["windows"] = args.window_size


## shuufle
indexes = np.arange(len(train_X))
np.random.shuffle(indexes)
train_X = train_X[indexes]
train_y = train_y[indexes]
log_dic["data"]["seed"] = 42

####################################################################################
# extract vmd feature
####################################################################################

if args.vmd: # 增加vmd特征提取

    alpha = 2000       # moderate bandwidth constraint
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)
    K = 3              # 3 modes
    DC = 0             # no DC part imposed
    init = 1           # initialize omegas uniformly
    tol = 1e-7

    vmd_train_X = []   # 存储vmd数据
    vmd_test_X = []    
    for i in range(len(train_X)): # for every one example
        vmd_compose_matrix , u_hat, omega = VMD(train_X[i], alpha, tau, K, DC, init, tol)
        vmd_train_X.append(vmd_compose_matrix.T) # append(shape=(W,3))

    for i in range(len(test_X)): # for every one example
        vmd_compose_matrix , u_hat, omega = VMD(test_X[i], alpha, tau, K, DC, init, tol)
        vmd_test_X.append(vmd_compose_matrix.T) # append(shape=(W,3))
    
    train_X = np.expand_dims(train_X, axis=-1)
    test_X = np.expand_dims(test_X, axis=-1)
    train_X = np.concatenate((train_X,np.array(vmd_train_X)), axis=-1) # (N, W, 1) (N, W, 3) -> (B, W, 4)
    test_X = np.concatenate((test_X,np.array(vmd_test_X)), axis=-1)
    print(train_X.shape)
    print(test_X.shape)
    log_dic["data"]["vmd"] = 3
    dim = 4 ## 输入维度大小

else:
    train_X = np.expand_dims(train_X, axis=-1)
    test_X = np.expand_dims(test_X, axis=-1)
    dim = 1 ## 输入维度大小
    
log_dic['data']["train_x"] = str(train_X.shape)
log_dic['data']["train_y"] = str(train_y.shape)
log_dic['data']["test_x"] = str(test_X.shape)
log_dic['data']["test_y"] = str(test_y.shape)

####################################################################################
#   Build Model
####################################################################################

train_model = build_model(args.model, args.window_size, dim, TCN_layers=args.layer)
log_dic["model"]["name"] = args.model

optimizer = keras.optimizers.Adam(learning_rate=args.lr)
log_dic["model"]["lr"] = args.lr

train_model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
log_dic["model"]["loss"] = "mse"

# 绘制模型图
# plot_model(train_model, to_file=os.path.join(args.save, args.model+"_model.png"),
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True)


####################################################################################
#   Training Code
####################################################################################

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)
history = train_model.fit(x=train_X, y=train_y,
                    # train_dataset,
                    epochs=args.epoch, batch_size=args.batchsize, shuffle=True,
                    validation_split = 0.2,callbacks=[callback])

log_dic["traing"]["epoch"] = args.epoch
log_dic["traing"]["batchsize"] = args.batchsize
log_dic["traing"]["early_stop"] = len(history.history["val_loss"])

epoch = len(history.history["val_loss"])

# 保存训练好的模型
train_model.save(os.path.join(args.save, args.model+".h5"))

# 最佳验证集结果：用来挑选超参
val_loss = min(history.history['val_loss'])
val_epoch = history.history['val_loss'].index(val_loss) + 1
val_mae = history.history['val_mae'][val_epoch-1]
train_loss = history.history['loss'][-1]
train_mae = history.history['mae'][-1]
log_dic["traing"]["best_val_mse"] = round(val_loss,3)
log_dic["traing"]["best_val_mae"] = round(val_loss,3)
log_dic["traing"]["best_epoch"] = val_epoch
log_dic["traing"]["final_train_loss"] = round(train_loss,3)
log_dic["traing"]["final_train_mae"] = round(train_mae,3)
log_dic["traing"]["final_val_loss"] = round(history.history['val_loss'][-1],3)
log_dic["traing"]["final_val_mae"] = round(history.history['val_mae'][-1],3)

## display
# loss
plt.figure(figsize=(20,10)) 
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.title(args.model + 'train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig(os.path.join(args.save, 'train_loss.jpg'))
if plot:
    plt.show()

####################################################################################
#   Testing Model
####################################################################################

train_model = keras.models.load_model(os.path.join(args.save, args.model+".h5"))
print("sucessfully load the model")

# 测试集评估模型
evaluate_result = train_model.evaluate(x=test_X, y=test_y)
print("Test loss:{}, mae:{}".format(evaluate_result[0],evaluate_result[1]))
log_dic["testing"]["evaluate"] = {"loss":evaluate_result[0], "mae":evaluate_result[1]}

# 分析预测结果
predicts = train_model.predict(test_X)
predicts_inv = scale.inverse_transform(predicts)
test_predicts = np.reshape(predicts, len(predicts))
test_predicts_inv = np.reshape(predicts_inv, len(predicts))

trues = test_y
trues_inv = scale.inverse_transform(test_y)
test_trues = np.reshape(trues, (len(trues)))
test_trues_inv = np.reshape(trues_inv, (len(trues_inv)))

r = rmse(test_trues,test_predicts)
m = mae(test_trues,test_predicts)
ms = mse(test_trues,test_predicts)
mape_t = mape(test_trues,test_predicts)
r_inv = rmse(test_trues_inv,test_predicts_inv)
m_inv = mae(test_trues_inv,test_predicts_inv)
mse_inv = mse(test_trues_inv,test_predicts_inv)
mape_inv = mape(test_trues_inv,test_predicts_inv)
predict_df = pd.DataFrame({
    "test_trues":test_trues,
    "test_predicts":test_predicts,
    "test_trues_inv":test_trues_inv,
    "test_predicts_inv":test_predicts_inv
})
predict_df.to_csv(os.path.join(args.save , 'predicts.csv'))

print("Test set:"+">"*5+args.model+"<"*5)
print("standard result")
print("rmse:",r)
print("mae:",m)
print("mse:",ms)
print("mape:",mape_t)
print("inverter result")
print("rmse:",r_inv)
print("mae:",m_inv)
print("mse:",mse_inv)
print("mape:",mape_inv)

metrics_df = pd.DataFrame({
    "method":["rmse","mae","mse","mape"],
    "standard":[r,m,ms,mape_t],
    "inverse":[r_inv,m_inv,mse_inv,mape_inv],
})
log_dic["testing"]["standard"] = {"rmse":r, "mse": ms, "mae": m, "mape": mape_t}
log_dic["testing"]["inverse"] = {"rmse":r_inv, "mse": mse_inv, "mae": m_inv, "mape": mape_inv}

metrics_df.to_csv(os.path.join(args.save,'test_metrics.csv'))

# predict
plt.figure(figsize=(20,10))
plt.title("standard predict: rmse:{:.2f} mse:{:.2f} mae:{:.2f} mape:{:.2f}".format(r,ms,m,mape_t))
plt.plot(test_trues,'b', label="true")
plt.plot(test_trues,'bo', markersize=3, label="true")
plt.plot(test_predicts,'r', label='predict')
plt.plot(test_predicts,'rx', markersize=5, label='predict')
plt.legend(loc='best')
plt.savefig(os.path.join(args.save, 'standard_predict.jpg'))
if plot:
    plt.show()

plt.figure(figsize=(20,10))
plt.title("inverse predict: rmse:{:.2f} mse:{:.2f} mae:{:.2f} mape:{:.2f}".format(r_inv,mse_inv,m_inv,mape_inv))
plt.plot(test_trues_inv,'b', label="true")
plt.plot(test_trues_inv,'bo', markersize=3, label="true")
plt.plot(test_predicts_inv,'r', label='predict')
plt.plot(test_predicts_inv,'rx', markersize=5, label='predict')
plt.legend(loc='best')
plt.savefig(os.path.join(args.save, 'inverse_predict.jpg'))
if plot:
    plt.show()

## 保存参数与结果
with open(os.path.join(args.save, 'log.json'),'w') as f:
    json.dump(log_dic,f,indent=4)

print("result dir in: ",os.path.join(args.save))
