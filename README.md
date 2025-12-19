### 所需环境
torch==1.2.0  

### 文件下载
训练所需的deeplab_mobilenetv2.pth和deeplab_xception.pth可在百度网盘中下载。     
链接: https://pan.baidu.com/s/1IQ3XYW-yRWQAy7jxCUHq8Q 提取码: qqq4  

####训练
1、本文使用VOC格式进行训练。 
    使用json_to_dataset将json文件转化为png标签
    删除VOCdevkit/VOC2007/ImageSets、VOCdevkit/VOC2007/JPEGImages
    将datasets/SegmentationClass复制到VOCdevkit/VOC2007/SegmentationClass
    将图片文件放在VOCdevkit/VOC2007/JPEGImages  
2、在训练前利用voc_annotation.py文件生成对应的txt。    
3、在train.py文件夹下面，选择自己要使用的主干模型和下采样因子。本文提供的主干模型有mobilenet和xception。下采样因子可以在8和16中选择。需要注意的是，预训练模型需要和主干模型相对应。   
4、注意修改train.py的num_classes为分类个数+1。    
5、运行train.py即可开始训练。  

### 预测步骤
运行predict.py      
