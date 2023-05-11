# 基于深度学习的电力负荷预测算法研究
> 作者：物联 192 梁显武    
> 指导老师: 曹忠

本代码为本人本科毕业设计的一部分。



## 基于TCN的电力负荷预测框架

TCN 不像图像卷积那样通过池化层扩大感受野，而是通过增大扩张因子以及增加层数对感受野进行扩大，这使得它能够接受更长的历史时序信息，从而降低预测的误差、提高准确率。通过残差连接能够使得网络的层数加深而不丢失准确性,这种跨层连接的结构使得信息可以在神经网络的不同层之间直接传递，而不会受到层数的限制，从而提高了神经网络的训练效率和准确性。TCN 的结构主要包括扩张因果卷积以及残差连接组成。

![扩张因果卷积](image\Dilated_Causal_Conv.png)
<center>图1 扩张因果卷积</center>

![基于TCN的电力负荷预测](image\TCN.png)
<center>图2 基于TCN的电力负荷预测</center>



## 代码依赖

* python3.8
* keras==2.6.0
* matplotlib==3.5.2
* numpy==1.19.4
* pandas==1.4.3
* tensorflow==2.6.0
  

可以使用以下命令安装依赖项:  
`pip install -r requirements.txt`



## 实验平台

CPU:Xeon(R) CPU E5-2620 v4 @ 2.10GHz  
GPU：NVIDIA TITAN V



## 数据

本文使用了一个名为 REFIT 的公开数据集。REFIT数据集是由斯特拉斯克莱德大学、拉夫堡大学和东安格利亚大学合作创建的，数据集包含了 2013 年至 2014 年期间在拉夫堡地区的 20 个家庭的负荷数据由于数据量过大，本文的工作仅仅基于House1的数据进行的。

本文所使用的原始数据可以在`REFIT_source`文件夹中找到，经过预处理（数据清洗）之后的数据可以在`REFIT_processed`文件夹中找到。然后在脚本`data_process.ipynb`中对数据进行了清洗工作。
> 注意代码库里面的数据是用压缩形式保存的.zip，使用前需要将压缩包解压到当前文件夹中。

![原始数据](image\SourceData.png)
<center>图3 原始数据</center>

![原始数据表](image\SourceDataTable.jpg)
<center>图4 原始数据部分展示</center>



## 复现

所有的训练设置都已经在`.sh`脚本文件中配置好了，例如`TCN_w6.sh`表示使用处理好的数据对TCN、LSTM、DNN模型进行训练和预测，最终生成的结果将存放在`tuning_w6`文件夹中。如果要复现结果，直接在命令行中输入以下命令。
1. `TCN_w6.sh`
2. `TCN_w12.sh`
3. `TCN_w24.sh`
4. `TCN_w48.sh`
5. `TCN_w96.sh`
6. `TCN_w192.sh`
7. `TCN_w384.sh`



## 代码使用

代码文件的功能解释如下：
1. `data_process.ipynb` : 数据处理代码
2. `model.py` : 模型构建代码
3. `tools.py` : 模型评估方法和数据归一化方法
4. `train_univariate.py`: 训练代码
5. `result.ipynb`:查看结果代码

模型训练的代码使用如下：  
`python train_univariate.py --data REFIT_processed/RAW_House1_1T_processed.csv --target Aggregate --window_size 6 --model TCN --layer 2 --patience 10 --save tuning_test/TCN_Aggregate --epoch 1000 --batchsize 128 --lr 0.008 --gpu 9
`  
参数解释：

1. --data 训练数据的文件路径
2. --target 预测目标曲线
3. --window_size 数据窗口大小
4. --model 模型选择
5. --layer TCN模型的层数
6. --patience 早停止策略的容忍度
7. --save 训练结果保存的文件夹
8. --epoch 训练迭代轮数
9. --batchsize 批次大小
10. --lr 学习率大小
11. --gpu GPU编号



## 结果

每个训练结果都会保存到--save指定的文件夹中，例如`tuning_w6/`。使用`.sh`脚本所产生的结果如下：
![训练结果表](image\ResultTable.jpg)

<center>图5 训练结果表</center>



TCN 的对更长的历史窗口的信息捕获能力表现得更为明显，随着窗口增加，精度呈现出下降趋势，在窗口大小为 386 时预测误差达到了最小。

![窗口-精度](image\line.jpg)

<center>图6 窗口大小对模型的影响</center>



历史窗口大小为 48时，在曲线上模型预测的具体表现。

![Aggregate曲线](image\Cure.jpg)

<center>图7 窗口为48时，Aggregate曲线</center>



![Appliance1曲线](image\Cure2.jpg)
<center>图8 窗口为48时Appliance1曲线</center><br>

## 参考论文
[Bai S, Kolter J Z, Koltun V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling[J]. arXiv preprint arXiv:1803.01271, 2018.](https://arxiv.org/abs/1803.01271)