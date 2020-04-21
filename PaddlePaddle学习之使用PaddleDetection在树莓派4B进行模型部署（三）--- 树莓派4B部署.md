[PaddlePaddle学习之使用PaddleDetection在树莓派4B进行模型部署（一）--- 项目环境搭建](https://blog.csdn.net/qq_42549612/article/details/104991557)
[PaddlePaddle学习之使用PaddleDetection在树莓派4B进行模型部署（二）--- 深度学习模型训练](https://blog.csdn.net/qq_42549612/article/details/104996034)
[PaddlePaddle学习之使用PaddleDetection在树莓派4B进行模型部署（三）--- 树莓派4B部署](https://blog.csdn.net/qq_42549612/article/details/104998329)

此项目已公开，包括数据集在内已经打包上传，欢迎Fork！传送门：[Paddle_ssd_mobilenet_v1_pascalvoc](https://aistudio.baidu.com/aistudio/projectdetail/331209)

本文将使用ssd_mobilenet_v1_voc算法，以一个例子为说明如何利用paddleDetection完成一个项目----从准备数据集到完成树莓派部署，项目用到的工具是百度的[AI Studio](https://aistudio.baidu.com/aistudio/index)在线AI开发平台和树莓派4B
全部资料已经都打包在这里（PaddleDetection、Paddle-Lite-Demo、Paddle-Lite、opt）
链接：https://pan.baidu.com/s/1IKT-ByVN9BaVxfqQC1VaMw 
提取码：mdd1 

## 预测库编译
Paddle-Lite目前支持三种编译的环境：
1. Docker 容器环境
2. Linux（推荐 Ubuntu 16.04）环境
3. Mac OS 环境

本次项目仅涉及到树莓派的ARMLinux环境编译，其他编译环境请参考[Paddle-Lite官方文档](https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html)

编译环境要求
gcc、g++、git、make、wget、python
cmake（建议使用3.10或以上版本）
官方安装流程如下：
```python
# 1. Install basic software
apt update
apt-get install -y --no-install-recomends \
  gcc g++ make wget python unzip

# 2. install cmake 3.10 or above
wget https://www.cmake.org/files/v3.10/cmake-3.10.3.tar.gz
tar -zxvf cmake-3.10.3.tar.gz
cd cmake-3.10.3
./configure
make
sudo make install
```
此环境树莓派应该是会有的，可以自行检查，没有的包安装上即可。
至此完成所有的编译环境配置。
将 Paddle-Lite 和 Paddle-Lite-Demo 移动至树莓派中，放在自己方便的目录下即可，在这里我的 Paddle-Lite 放在了 /home/pi/ 下，将 Paddle-Lite-Demo 放在了 /home/pi/Desktop/ 下，并且将 /home/pi/Paddle/Paddle-Lite/lite/tools/build.sh 加上执行权限
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320214243568.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70)
所有工作完成后，即可开始编译Paddle-Lite
```python
cd /home/pi/Paddle/Paddle-Lite
sudo ./lite/tools/build.sh \
  --build_extra=OFF \
  --arm_os=armlinux \
  --arm_abi=armv7hf \
  --arm_lang=gcc \
  tiny_publish
```
虽然树莓派4B已经是 ARMv8 的CPU架构，但官方系统为32位，还是需要使用ARMv7架构的编译方式
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320212753129.png)
编译结束，结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320214454890.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70)
## 文件结构搭建
整体文件结构如下：
```
 object_detection_demo
        
  		Paddle-Lite：	
				include (编译好的Paddle—Lite的头文件)
				libs（存放armv7hf）
		                armv7hf（编译好的Paddle—Lite的库文件）
  		code：
				models(模型文件：model.nb)
				images(测试图片)
				CMakeLists.txt
				mask_detection.cc
				run.sh
```
1. 打开 /home/pi/Desktop/Paddle-Lite-Demo/PaddleLite-armlinux-demo/object_detection_demo 文件夹，在此目录下新建 Paddle-Lite、code 文件夹
2. Paddle-Lite文件夹下新建 include、libs 文件夹
3. libs文件夹下新建 armv7hf 文件夹
4. 将 images、labels、CMakeLists.txt、run.sh、object_detection_demo.cc 文件移入 code 文件夹下


对于 Paddle-Lite 的编译结果，我们需要使用的东西在 /home/pi/Paddle/Paddle-Lite/build.lite.armlinux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx 文件夹下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320214640752.png)

将 include 和 lib 中的头文件和库文件提取出来，分别放入 include 和 armv7hf 文件夹中
至此已做好文件结构的搭建


## 模型部署
接下来就是最后一步了，将模型放进文件中，稍作修改就大功告成了！
1. 进入 code 文件夹
2. 修改 labels 文件夹下的 pascalvoc_label_list ，内容必须与训练时的 label_list.txt 文件内容一致  （注意 pascalvoc_label_list 是纯文本文档，不是 .txt 文本文档，弄错了预测出来的框选标签会打 unknow 的！）
3. 将在[PaddlePaddle学习之使用PaddleDetection在树莓派4B进行模型部署（二）----- 深度学习模型训练](https://blog.csdn.net/qq_42549612/article/details/104996034)得到的 model.nb 放进 models 文件夹
4. 打开 run.sh 文件，注释掉第四行的 TARGET_ARCH_ABI=armv8 ，打开第五行的，取消第5行 TARGET_ARCH_ABI=armv7hf 的注释
5. 修改第六行的 PADDLE_LITE_DIR 索引到文件中Paddle-Lite目录
6. 修改第十九行的model文件的模型索引目录和预测图片的索引目录

```python
#!/bin/bash

# configure
#TARGET_ARCH_ABI=armv8 # for RK3399, set to default arch abi
TARGET_ARCH_ABI=armv7hf # for Raspberry Pi 3B
PADDLE_LITE_DIR=/home/pi/Desktop/Paddle-Lite-Demo/PaddleLite-armlinux-demo/object_detection_demo/Paddle-Lite
if [ "x$1" != "x" ]; then
    TARGET_ARCH_ABI=$1
fi

# build
rm -rf build
mkdir build
cd build
cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} -DTARGET_ARCH_ABI=${TARGET_ARCH_ABI} ..
make

#run
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./object_detection_demo ../models/model.nb ../labels/pascalvoc_label_list ../images/2001.jpg ./result.jpg

```
修改完run.sh文件后，就算是完成了所有的配置内容，可以开始放心的 RUN 了！！
```python
/home/pi/Desktop/Paddle-Lite-Demo/PaddleLite-armlinux-demo/object_detection_demo/code
sudo ./run.sh
```
最后的输出结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320225740273.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70)
图片的预测结果就是这样了，虽然一个类别只有300张图，但是总的来说结果还算不错！
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200320225748478.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70)
注：如果有用 Opencv-4.1.0 版本的，可能在编译 object_detection_demo.cc 时在 267、268 行会报错
源代码如下：

```cpp
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
```

`    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);`
由于新版本的API发生了变化。需要修改为如下代码：

```cpp
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
```
至此，整个项目就完成了，目前部署的是静态图的检测，后面会继续更新关于视频流的检测方法！
（C++懂得并不多的小白还在改预测代码中，等跑通后再发出来~）

----
参考资料
- 
[Paddle-Lite官方文档](https://paddle-lite.readthedocs.io/zh/latest/index.html)
