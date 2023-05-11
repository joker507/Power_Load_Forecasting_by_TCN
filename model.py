import tensorflow as tf
import tensorflow.keras as keras
from keras.utils.vis_utils import plot_model

# TCN残差连接块
def TCN_Block(inputs, dilation_rate, in_channels,
            out_channels, padding, kernel_size=2,dropout_rate=0.2):
    
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
    assert padding in ["causal","same"]

    # block1
    conv1 = keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
    batch1 = keras.layers.BatchNormalization(axis=-1)
    ac1 = keras.layers.Activation("relu")
    drop1 = keras.layers.Dropout(rate=dropout_rate)

    # block2
    conv2 = keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
    batch2 = keras.layers.BatchNormalization(axis=-1)
    ac2 = keras.layers.Activation("relu")
    drop2 = keras.layers.Dropout(rate=dropout_rate)

    # 输出维度一致
    downsample = keras.layers.Conv1D(filters=out_channels, 
        kernel_size=1, 
        padding="same", 
        kernel_initializer=init)
    
    ac3 = keras.layers.Activation("relu")

    # forward
    # inputs =  keras.Input(shape=(window_size, in_channels))
    x = conv1(inputs)
    x = batch1(x)
    x = ac1(x)
    x = drop1(x)
    x = conv2(x)
    x = batch2(x)
    x = ac2(x)
    y = drop2(x)
    if in_channels != out_channels:
        res = downsample(inputs)
    else:
        res = inputs

    outputs = ac3(keras.layers.add([y,res]))

    return outputs

# TCN
def TCN(num_inputs=10 ,num_channels=[150,150,150,150], 
           window_size=6, kernel_size=2, dropout=0.2):

    # input layers
    inputs = keras.Input(shape=(window_size,num_inputs))
    t = inputs
    
    # TCN layers
    num_levels = len(num_channels) # 层数
    for i in range(num_levels):
        dilation_size = 2 ** i
        in_channels = num_inputs if i == 0 else num_channels[i-1]
        out_channels = num_channels[i]
        # tcn clock
        t = TCN_Block(t, 
            dilation_rate=dilation_size, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            padding='causal', 
            kernel_size=kernel_size, 
            dropout_rate=dropout)
    
    # output layers
    # outputs = keras.layers.Conv1D(filters=1, kernel_size=3)(outputs)
    # outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(1,activation=None)(t[:,-1,:]) # (batsize,final_time_step,dim)

    tcn_model = keras.Model(inputs=inputs, outputs=outputs)
    tcn_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    tcn_model.summary()
    return tcn_model

# LSTM
def LSTM(window_size,units=12,dim=10):
    x = keras.layers.Input(shape=(window_size, dim))
    t = keras.layers.LSTM(units=units, return_sequences=False)(x) 
    y = keras.layers.Dense(1)(t)

    model = keras.Model(inputs=x,outputs=y)
    model.summary()
    return model

# GRU
def GRU(window_size):
    x = keras.layers.Input(shape=(window_size, 10))
    t = keras.layers.GRU(units=12, return_sequences=True)(x) 
    t = keras.layers.Dropout(0.2)(t)
    t = keras.layers.GRU(units=12, return_sequences=False)(t) 
    t = keras.layers.Dropout(0.2)(t)
    y = keras.layers.Dense(1)(t)

    model = keras.Model(inputs=x,outputs=y)
    # model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model  

# ANN（DNN）
def ANN(window_size,dim=10):
    x = keras.layers.Input(shape=(window_size,dim))
    t = keras.layers.Flatten()(x)
    t = keras.layers.Dense(128, activation='sigmoid')(t)
    t = keras.layers.Dense(256, activation='sigmoid')(t)
    t = keras.layers.Dense(128, activation='sigmoid')(t)
    y = keras.layers.Dense(1)(t)
    
    model = keras.Model(inputs=x,outputs=y)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model

## 模型构建
def build_model(model_name, window_size,dim=10, TCN_layers=2):
    model_type = model_name
    if model_type == "LSTM":
        train_model = LSTM(window_size=window_size,units=12,dim=dim)
    elif model_type == "ANN":
        train_model = ANN(window_size,dim)
    elif model_type == "GRU":
        train_model = GRU(window_size,units=12)
    elif model_type == "TCN":
        num_inputs=1 
        num_channels=[20 for i in range(TCN_layers)]
        print("TCN_layers is ", TCN_layers)
        kernel_size=3
        dropout=0.2
        train_model = TCN(num_inputs,num_channels,window_size,kernel_size,dropout)
    else:
        assert False, "请选择正确的模型"

    return train_model

if __name__ == "__main__":
    window_size = 6
    m = ANN(window_size,dim=1)
    plot_model(m, 'model.png', show_shapes=True, show_layer_names=True)

    
