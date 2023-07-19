# 找到想进行测试的model
epoch = "0"

import os
import yaml

save_root = "output"
path_allEpochs_modelsTest = os.path.join(save_root,"allEpochs_modelsTest")
path_all_models_results = os.path.join(path_allEpochs_modelsTest, "all_models_results.txt")
path_all_models_results_yaml = os.path.join(path_allEpochs_modelsTest, "all_models_results.yaml")

import torch
from torch.utils.data import DataLoader
from utils import LoadData, WriteData
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2
from tqdm import tqdm
from matplotlib import pyplot as plt

def test(dataloader, model, device):
    pred_list = []
    # 将模型转为验证模式，只需要前向传播
    model.eval()
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in tqdm(dataloader):
            # 将数据转到GPU
            X, y = X.to(device), y.to(device)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            pred_softmax = torch.softmax(pred, 1).cpu().numpy()
            pred_list.append(pred_softmax.tolist()[0])
        return pred_list





if __name__=="__main__":
    pathConfigYaml = os.path.join("information", "config.yaml")
    with open(pathConfigYaml, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    myNet = config["myNet"]
    dataset_label = config["dataset_label"]
    classNum = len(dataset_label)
    epochs = config["epochs"]

    model_name = myNet + "_" + config["mark"]
    # 查看保存了测试结果的yaml文件
    with open(path_all_models_results_yaml, 'r', encoding='utf-8') as file:
        all_models_results = yaml.safe_load(file)
    print(all_models_results)

    f"output\\modelBackUp{model_name}_epoch{epoch}.pth"



