## 代码使用

1. 准备数据集，在data中有将avi视频文件转换为图像帧的data_pro.py 
2. 将训练以及预测脚本中的`num_classes`设置成你数据的类别数
3. 在`train.py`脚本中将`--data-path`设置成待训练数据的路径
4. 下载预训练权重，在`model.py`文件中每个模型都有提供预训练权重的下载地址，根据使用的模型下载对应预训练权重
5. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
6. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
7. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
8. 在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片路径
9. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict.py`脚本进行预测了
