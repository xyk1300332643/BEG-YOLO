
import torch
from ultralytics import YOLO
import numpy as np

# 加载预训练的 YOLOv8n 模型
model = YOLO(r'D:\PyCharm 2021.1.3.rar.baiduyun.p\workspace\yolov8\ultralytics-main\runs\detect\train\weights\best.pt')


results = model.predict(source=r'D:\PyCharm 2021.1.3.rar.baiduyun.p\workspace\zoo_data\train\images\001.jpg',save=True,save_conf=True,save_txt=True,name='output')

name_dict = results[0].names
print(name_dict)
# 查看结果
for r in results:
    print(r.probs)  # 打印包含检测到的类别概率的Probs对象

#source后为要预测的图片数据集的的路径
#save=True为保存预测结果
#save_conf=True为保存坐标信息
#save_txt=True为保存txt结果，但是yolov8本身当图片中预测不到异物时，不产生txt文件