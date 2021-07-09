import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as data
# from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from time import time
from tqdm import tqdm

from model import DeepSets
from dataset import MNIST_train_load

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def make_dataloader(X, y, batch_size):
    """ Dataloader作成 """

    dataset = data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def train(model, X_train, y_train, X_val, y_val, optimizer, criterion):
    """ 学習 """

    # ネットワークをGPUへ
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.train() # 訓練モードに
    dataloader = make_dataloader(X_train, y_train, 100)

    # データローダーからミニバッチを取り出すループ
    for inputs, labels in dataloader:

        inputs, labels = inputs.to(device), labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # 順伝搬（forward）計算
        outputs = model(inputs)

        loss = criterion(outputs.reshape(-1), labels)  # 損失を計算
        # loss = criterion(outputs, labels)  # 損失を計算

        # 逆伝播（backward）計算
        loss.backward()
        optimizer.step()

    model.eval() # 推論モードに

    # f1 macro算出
    outputs = model(torch.from_numpy(X_val).to(device))
    preds = torch.round(outputs.reshape(-1))
    # _, preds = torch.max(outputs, 1)
    f1_macro = f1_score(y_val, preds.to('cpu').detach().numpy().astype(int), average="macro")

    return -f1_macro

def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # Adam
    if optimizer_name == optimizer_names[0]: 
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)

    # MomentumSGD
    elif optimizer_name == optimizer_names[1]:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)

    # RMSprop
    else:
        optimizer = optim.RMSprop(model.parameters())

    return optimizer    

def objective(trial):

    EPOCH = 5 # 学習試行数
    num_folds = 4

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデル定義
    model_list = [DeepSets(patch_size=28, feature_n=2, output_n=1, pool_mode="sum").to(device) for i in range(num_folds)]

    # 最適アルゴリズム
    # optimizer = get_optimizer(trial, model)
    optimizer_list = [get_optimizer(trial, model_list[i]) for i in range(num_folds)]

    # 損失関数
    criterion = nn.MSELoss()

    ### 学習（StratifiedKFoldを使用）#################################
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    f1_list = np.zeros(num_folds)

    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):

        print("--- fold {} ------------".format(fold))

        X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]

        # 学習
        for step in tqdm(range(EPOCH)):
            f1_list[fold] = train(model_list[fold], X_train, y_train, X_val, y_val, optimizer_list[fold], criterion, EPOCH)

    return np.mean(f1_list)

if __name__ == '__main__':

    # データ前処理
    X, y = MNIST_train_load()
    X, y = X.to('cpu').detach().numpy().copy(), y.to('cpu').detach().numpy().copy()

    # チューニング開始
    start = time()

    TRIAL_SIZE = 10 # チューニング試行数
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)
    print(study.best_params)

    end = time()
    print("time: {}m".format((end-start)/60))