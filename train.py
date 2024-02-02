import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import os,sys
sys.path.append(os.getcwd())
sys.path.append(r'/opt/data/private/ODIR_2024/')

from dataset.MyDataSet import ODIRDataSet
from utils.augment import train_transform, val_test_transform
from model.MyModel import ResNet18, ResNet50, Inceptionv3, VGG19
from utils.loss_weight_cal import pos_weight
from utils.metrics_cal import calculate_metrics
from utils.metrics_show import show_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print(">>>===========device:{}({},{}MB)".format(device,GPU_device.name,GPU_device.total_memory / 1024 ** 2))

# 超参数
parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str)
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument("--num_class", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--threshold", type=float, default=0.5)

args = parser.parse_args()
model_name = args.model_name
epoches = args.epoches
batch_size = args.batch_size
num_class = args.num_class
lr = args.lr
dropout_prob= args.dropout_prob
weight_decay = args.weight_decay # L2
threshold = args.threshold

print("\n>>>==========================<<<\n")
print(f"Paramaters of {model_name}:\n")
print("epochs:{} batch_size:{} learning_rate:{} dropout_prob:{} weight_decay:{} threshold:{}".format(epoches,batch_size,lr,dropout_prob,weight_decay,threshold))
print("\n>>>==========================<<<\n")

# 读取CSV文件
csv_path = "dataset/ODIR-5K_Annotations.csv"
df = pd.read_csv(csv_path)

# 划分训练集、验证集和测试集
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

# 加载数据
train_ds = ODIRDataSet(train_df, transform=train_transform)
val_ds = ODIRDataSet(val_df, transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# 定义模型
model = Inceptionv3(num_class, dropout_prob).to(device)

# 定义损失函数和优化器
loss_weight = pos_weight
loss_weight = torch.tensor(loss_weight).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay,nesterov=True)

# 训练
f1_best = 0
loss_history = {'train': [], 'val': []}  # 用于保存每个epoch的训练集和验证集损失值
for epoch in tqdm(range(epoches)):
    model.train()
    total_loss_tr = 0

    for iteration, (img1, img2, labels) in enumerate(train_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        output1 = output1.logits
        output2 = output2.logits

        loss_l = criterion(output1, labels)
        loss_r = criterion(output2, labels)
        loss = loss_l + loss_r
        total_loss_tr += loss.item()
        loss.backward()
        optimizer.step()

    average_loss_tr = total_loss_tr/len(train_loader)

    # 验证
    model.eval()
    total_loss_val = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for iteration, (img1, img2, labels) in enumerate(val_loader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            output1, output2 = model(img1, img2)

            loss_l = criterion(output1, labels)
            loss_r = criterion(output2, labels)
            loss = loss_l + loss_r
            total_loss_val += loss.item()

            predictions = (output1.sigmoid() + output2.sigmoid()) / 2.0
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        average_loss_val = total_loss_val/len(val_loader)
        loss_history['train'].append(average_loss_tr)
        loss_history['val'].append(average_loss_val)

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        acc, precision, recall, f1, num = calculate_metrics(all_predictions, all_labels, threshold)

        print("[Train Loss:{:.4f} Val Loss:{:.4f}]".format(average_loss_tr,average_loss_val))
        average_acc, average_f1 = show_metrics(acc, precision, recall, f1, num)

        if average_f1 >= f1_best:
            torch.save(model.state_dict(), f'./checkpoint/{model_name}_best.pth')
            print(f"A new record has been saved.")
            f1_best = average_f1


        df = pd.DataFrame(loss_history)
        df.to_csv(f'./logs/loss_{model_name}.txt', index=False)