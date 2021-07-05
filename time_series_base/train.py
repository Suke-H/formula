# パッケージのimport
from os import replace
import numpy as np
from tqdm import tqdm
import glob 
import re
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

from dataset import MNIST_load
from model import DeepSets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def train(model, dataloader, optimizer, criterion, num_epochs, model_path="models/"):
    model = model.to(device)
    model.train()
    print("Epoch: 1")
    
    for epoch in tqdm(range(1, num_epochs+1)):
        correct = 0
        epoch_loss = 0
        total = 0
 
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            # print(outputs)
            # print(labels)
            # loss = criterion(outputs.reshape, labels)
            loss = criterion(outputs.reshape(-1), labels) # 回帰
            loss.backward()
            optimizer.step()
    
            outputs = torch.round(outputs) # 回帰
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            epoch_loss += loss
        
        if epoch % 10 == 0:
            print("Train Acc : %.4f" % (correct/total))
            print("Train Loss : %.4f" % (epoch_loss/total))
            print("Epoch: {}".format(epoch))

        if epoch % 100 == 0:
            torch.save(model.state_dict(), model_path+str(epoch)+".pth")

def test(model, test_loader, out_path="data/test_outputs"):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    y = []
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            outputs = torch.round(outputs) # 回帰
            y.extend(outputs.reshape(-1).to('cpu').detach().numpy().copy().tolist())

            # _, predicted = torch.max(outputs.data, 1)
            # correct += (predicted == labels).sum().item()
            # total += labels.size(0)

            # predicted = predicted.to('cpu').detach().numpy().copy().tolist()
            # y.extend(predicted)
            
 
    # print("Test Acc : %.4f" % (correct/total))

    np.save(out_path, np.array(y))

def display(terms, ys, t, output, epoch, out_path):

    #表示領域を設定（行，列）
    fig, ax = plt.subplots(2, 3)  

    #図を配置
    for i, (term, y) in enumerate(zip(terms, ys)):
        plt.subplot(2, 3, i+1)
        title = str(y)
        plt.title(title, fontsize=10)    #タイトルを付ける
        plt.tick_params(color='white')      #メモリを消す
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.imshow(term.reshape(28,28), cmap="gray")   #図を入れ込む

    plt.subplot(2, 3, 6)
    x, y = 0.5, 0.5
    text="target="+str(t)+",output="+str(output)
    plt.text(x, y, text, horizontalalignment='center')

    #図が重ならないようにする
    plt.tight_layout()

    #保存
    plt.savefig(out_path + "display2/"+str(epoch)+".png")
    plt.close()

if __name__ == "__main__":
    train_loader, val_loader, test_loader, test_x, test_y = MNIST_load()
    # model = DeepSets(patch_size=28, feature_n=2, output_n=46, pool_mode="sum")
    model = DeepSets(patch_size=28, feature_n=2, output_n=1, pool_mode="sum")
    optimizer = optim.Adam(model.parameters(), lr=10**(-3))
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    num_epochs = 1000
    model_path = "models2/"
    # train(model, train_loader, optimizer, criterion, num_epochs, model_path=model_path)

    model_paths = sorted(glob.glob(model_path + "**"), key=numericalSort, reverse=True)
    print(model_paths[0])
    model.load_state_dict(torch.load(model_paths[0], map_location=torch.device('cpu')))

    test(model, test_loader)

    ######
    # test_x = np.array([test_dataset[i][0].cpu().numpy() for i in range(10000)])
    # test_y = np.array([test_dataset[i][1] for i in range(10000)])
    term_indices = np.load("data/test.npy")
    group_x = []
    group_t = []
    group_sum = []

    for terms in term_indices:
        terms_x = test_x[terms, :, :, :]
        terms_t = test_y[terms]
        sum_t = np.sum(terms_t)

        group_x.append(terms_x)
        group_t.append(terms_t)
        group_sum.append(sum_t)

    group_x = np.array(group_x)
    group_t = np.array(group_t)
    group_sum = np.array(group_sum)

    ######
    outputs = np.load("data/test_outputs.npy")
    out_path = "data/"

    display_indices = np.random.choice(10000, 50, replace=False)
    for idx in display_indices:
        display(group_x[idx, :, :, :, :], group_t[idx], group_sum[idx], outputs[idx], idx, out_path)
