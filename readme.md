# 使用方法
## 准备环境
- Anaconda 3.7
- Pytorch 1.7（及其对应的torchvision）
- scikit-learn（sklearn）
- numpy

## 准备数据集（**注意：目前仅支持coco2017**）
- 在`根`目录中创建`data`文件夹，并将数据集存放在`data`文件夹中，并确保目录结构为：
```
├─data
│  ├─annotations
│  ├─test2017
│  ├─train2017
│  └─val2017
```
## 聚类（**注意：在训练、评估和推断前，需要先执行此操作**）
- 使用以下命令，或是使用vscode的调试配置：`聚类产生intra classes`
```
python intra_classes_clustering.py
```
## 训练（使用聚类标签）
- 使用以下命令，或是使用vscode的调试配置：`训练（使用聚类标签）`  
- 生成的日志位于`logs`文件夹下  
- 生成的模型位于`checkpoint`文件夹下
```
python train.py --image_min_side=400 --image_max_side=600 --learning_rate=0.001 --workers=4 --epochs=25 --num_steps_to_display=500 -b=backbone/resnet50-19c8e357.pth --intra_class
```
## 训练（使用原始标签）
- 使用以下命令，或是使用vscode的调试配置：`训练（使用原始标签）`  
- 生成的日志位于`logs`文件夹下  
- 生成的模型位于`checkpoint`文件夹下
```
python train.py --image_min_side=400 --image_max_side=600 --learning_rate=0.001 --workers=4 --epochs=25 --num_steps_to_display=500 -b=backbone/resnet50-19c8e357.pth 
```
## 评估（使用聚类标签）
- 使用以下命令，或是使用vscode的调试配置：`评估（使用聚类标签）`  
- 生成的标签位于`根`目录下
```
python eval.py --image_min_side=400 --image_max_side=600 --workers=4 --backbone_path=backbone/resnet50-19c8e357.pth --model_path=checkpoint/checkpoint-intra-class.pth
```
## 推断（使用聚类标签）
- 使用以下命令，或是使用vscode的调试配置：`推断（使用聚类标签）`  
- 生成的推断结果位于`outputs`目录下
```
python infer.py --image_min_side=400 --image_max_side=600 --backbone_path=backbone/resnet50-19c8e357.pth --model_path=checkpoint/checkpoint-intra-class.pth
```

