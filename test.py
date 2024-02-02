import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
import os,sys
sys.path.append(os.getcwd())
sys.path.append(r'/opt/data/private/ODIR_2024/')

from dataset.MyDataSet import ODIRDataSet
from utils.augment import train_transform, val_test_transform
from model.MyModel import ResNet18, ResNet50, Inceptionv3
from utils.metrics_cal import calculate_metrics
from utils.metrics_show import show_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print(">>>===========device:{}({},{}MB)".format(device,GPU_device.name,GPU_device.total_memory / 1024 ** 2))

# 超参数
parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument("--num_class", type=int, default=8)
parser.add_argument("--threshold", type=float, default=0.5)

args = parser.parse_args()
model_name = args.model_name
batch_size = args.batch_size
num_class = args.num_class
threshold = args.threshold

print("\n>>>==========================<<<\n")
print(f"HyperParamaters of test {model_name}:\n")
print("batch_size:{} threshold:{}".format(batch_size, threshold))
print("\n>>>==========================<<<\n")

# 读取CSV文件
csv_path = "dataset/ODIR-5K_Annotations.csv"
df = pd.read_csv(csv_path)

# 划分训练集、验证集和测试集
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# 加载数据
test_ds = ODIRDataSet(test_df, transform=val_test_transform)

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# 定义模型加载参数
model = Inceptionv3(num_class, dropout_prob=0).to(device)
checkpoint = torch.load(f'./checkpoint/{model_name}.pth', map_location=device)
model.load_state_dict(checkpoint)

# 测试
model.eval()
with torch.no_grad():
    all_predictions = []
    all_labels = []
    for iteration, (img1, img2, labels) in enumerate(test_loader):
        if iteration % (len(test_loader)/3) == 0:
            print(f"===Testing {iteration}/{len(test_loader)}")
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        output1, output2 = model(img1, img2)

        predictions = (output1.sigmoid() + output2.sigmoid()) / 2.0
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc, precision, recall, f1, num = calculate_metrics(all_predictions, all_labels, threshold)
    average_acc, average_f1 = show_metrics(acc, precision, recall, f1, num)
