from ultralytics import YOLO
import torch
import torch.cuda
if __name__ == '__main__':

    #从头开始训练一个全新的YOLO模型
    model = YOLO('./xx-YOLO/EMA-YOLO_yolov8.yaml')

    #加载预训练的YOLO模型（推荐用于训练）
    #model = YOLO('./YOLOv8_parameter/yolov8x.pt')

    #model = YOLO('./xx-YOLO/BGF-YOLO_yolov8.yaml').load('./YOLOv8_parameter/yolov8x.pt')  # 从YAML建立并转移权重
    torch.cuda.empty_cache()
    
    results = model.train(data='./NEU-DET_data.yaml',epochs = ,batch=8,patience=20)
    # train the model#imgsz保存的尺寸大小
    # #patience是指在验证损失不再下降时，模型继续训练的轮数2。如果在patience轮后，验证损失仍然没有改善，那么模型就会停止训练，以避免过拟合
    #评估模型在验证集上的性能
    #workers默认是8，是加载数据的所用的线程数量
    results = model.val()

    #使用模型对图片进行目标检测
    #results = model('路径')

    # export
    #success = model.export(format = 'onnx',batch=1)
