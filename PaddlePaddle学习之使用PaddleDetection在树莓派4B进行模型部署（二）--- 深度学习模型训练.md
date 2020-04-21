[PaddlePaddle学习之使用PaddleDetection在树莓派4B进行模型部署（一）--- 项目环境搭建](https://blog.csdn.net/qq_42549612/article/details/104991557)
[PaddlePaddle学习之使用PaddleDetection在树莓派4B进行模型部署（二）--- 深度学习模型训练](https://blog.csdn.net/qq_42549612/article/details/104996034)
[PaddlePaddle学习之使用PaddleDetection在树莓派4B进行模型部署（三）--- 树莓派4B部署](https://blog.csdn.net/qq_42549612/article/details/104998329)

此项目已公开，包括数据集在内已经打包上传，欢迎Fork！传送门：[Paddle_ssd_mobilenet_v1_pascalvoc](https://aistudio.baidu.com/aistudio/projectdetail/331209)

本文将使用ssd_mobilenet_v1_voc算法，以一个例子为说明如何利用paddleDetection完成一个项目----从准备数据集到完成树莓派部署，项目用到的工具是百度的[AI Studio](https://aistudio.baidu.com/aistudio/index)在线AI开发平台和树莓派4B
全部资料已经都打包在这里（PaddleDetection、Paddle-Lite-Demo、Paddle-Lite、opt）
链接：https://pan.baidu.com/s/1IKT-ByVN9BaVxfqQC1VaMw 
提取码：mdd1 
## 环境配置
```python
#安装Python依赖库
!pip install -r requirements.txt
```
```python
#测试项目环境
!export PYTHONPATH=`pwd`:$PYTHONPATH
!python ppdet/modeling/tests/test_architectures.py
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320184242175.png)
出现 No module named 'ppdet' 是环境配置的问题，有两种解决办法：
1. 设置环境变量
```python
%env PYTHONPATH=/home/aistudio/PaddleDetection
```
2. 找到报错的文件添加以下代码

```python
import sys
DIR = '/home/aistudio/PaddleDetection'
sys.path.append(DIR)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320200600156.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70)
测试环境通过后，就可以开始训练了
## 开始训练
训练命令如下：

```python
%cd home/aistudio/PaddleDetection/
!python -u tools/train.py -c configs/ssd/ssd_mobilenet_v1_voc.yml --use_tb=True --eval
```
训练完成后输出的模型保存在 ./PaddleDetection/output/ssd_mobilenet_v1_voc 文件夹下，本次训练总轮数默认为28000轮，每隔2000轮保存一次模型，以轮次命名的均为阶段性模型，model_final为训练结束时保存的模型，best_model是每次评估后的最佳mAP模型

```python
#测试，查看模型效果
%cd home/aistudio/PaddleDetection/
!python tools/infer.py -c configs/ssd/ssd_mobilenet_v1_voc.yml --infer_img=/home/aistudio/2001.jpg#infer_img输入需要预测图片的路径，看一下效果
```
## 模型转换
接下来，需要将原生模型转化为预测模型

```python
!python -u tools/export_model.py -c configs/ssd/ssd_mobilenet_v1_voc.yml --output_dir=./inference_model_final
```
生成的预测模型保存在 ./PaddleDetection/inference_model_final/ssd_mobilenet_v1_voc 文件夹下，会生成两个文件，模型文件名和参数文件名分别为__model__和__params__

由于部署到树莓派4B上需要使用Paddle-Lite，而PaddlePaddle的原生模型需要经过[opt](wget%20https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/opt)工具转化为Paddle-Lite可以支持的naive_buffer格式

```python
%cd /home/aistudio/
#复制opt文件到相应目录下
!cp opt /home/aistudio/PaddleDetection/inference_model_final/ssd_mobilenet_v1_voc
#进入预测模型文件夹
%cd /home/aistudio/PaddleDetection/inference_model_final/ssd_mobilenet_v1_voc
#下载opt文件
#!wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/opt
#给opt加上可执行权限
!chmod +x opt
#使用opt进行模型转化,将__model__和__params__转化为model.nb
!./opt --model_file=__model__ --param_file=__params__ --optimize_out_type=naive_buffer   --optimize_out=./model
!ls
```
这个opt自己下载实在是太慢了，因此我在网盘里已经准备好了opt文件，可以直接上传至AI Studio操作，最终结果如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320204424262.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70)
到目前为止，在AI Studio上的所有内容已经完成，文章（一）（二）的目的就是为了生成这个model.nb文件，将其部署在树莓派4B上使用。

------
参考资料
- 
[PaddleDetection官方文档](https://github.com/PaddlePaddle/PaddleDetection)
[Paddle-Lite官方文档 — C++Demo](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/cpp_demo.html)
