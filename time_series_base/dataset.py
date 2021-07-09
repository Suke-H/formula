
import numpy as np
import os
from matplotlib import pylab as plt

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageTransform():
    """
    画像の前処理クラス。
    torchテンソル化と標準化を行う。
    """

    def __init__(self):
        self.data_transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5, ), (0.5, ))])

    def __call__(self, img):
        return self.data_transform(img)

class Subset(torch.utils.data.Dataset):
    """
    インデックスを入力にデータセットの部分集合を取り出す

    Arguments:
        dataset : 入力データセット
        indices : 取り出すデータセットのインデックス
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)        

def make_summation_dataset(N, n_terms, type, x, t):
    
    # 組み合わせの仕方を保存していたら読み込み
    if os.path.exists("data/" + type + ".npy"):
        print("Combination list exists. Now loading...")
        term_indices = np.load("data/" + type + ".npy")

    #保存していなければ，無作為抽出で足し算データセット用のインデックス作成
    else:
        print("Combination list was not found. Now creating...")
        term_indices = np.random.choice(len(t), size=(N, n_terms), replace=True)
        np.save("data/" + type, term_indices)

    group_x = []
    group_sum = []

    for terms in term_indices:
        terms_x = x[terms, :, :, :]
        terms_t = t[terms]
        sum_t = np.sum(terms_t)

        group_x.append(terms_x)
        group_sum.append(sum_t)

    group_x = torch.Tensor(np.array(group_x)).to(device)
    # group_sum = torch.Tensor(np.array(group_sum)).long().to(device)
    group_sum = torch.Tensor(np.array(group_sum)).to(device)

    return group_x, group_sum

def MNIST_load():
    
    # 前処理用の関数
    transform = ImageTransform()
    img_transformed = transform

    # データセット読み込み + 前処理
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                        train=True, download=True, transform=img_transformed)
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                    train=False, download=True, transform=img_transformed)

    # numpyに変換
    train_x = np.array([train_dataset[i][0].cpu().numpy() for i in range(60000)])
    train_y = np.array([train_dataset[i][1] for i in range(60000)])
    test_x = np.array([test_dataset[i][0].cpu().numpy() for i in range(10000)])
    test_y = np.array([test_dataset[i][1] for i in range(10000)])

    # 足し算データセット作成
    group_train_x, group_train_y = make_summation_dataset(60000, 5, "train", train_x, train_y)
    group_test_x, group_test_y = make_summation_dataset(10000, 5, "test", test_x, test_y)

    # datasetオブジェクト作成
    trainval_dataset = torch.utils.data.TensorDataset(group_train_x, group_train_y)
    test_dataset = torch.utils.data.TensorDataset(group_test_x, group_test_y)

    # trainデータセットをtrain/valに分割
    n_samples = len(trainval_dataset) # n_samples is 60000
    train_size = int(n_samples * 0.8) # train_size is 48000

    subset1_indices = list(range(0,train_size)) # [0,1,.....47999]
    subset2_indices = list(range(train_size,n_samples)) # [48000,48001,.....59999]

    train_dataset = Subset(trainval_dataset, subset1_indices)
    val_dataset = Subset(trainval_dataset, subset2_indices)

    # dataloaderオブジェクト作成
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader, test_x, test_y

def MNIST_train_load():
    
    # 前処理用の関数
    transform = ImageTransform()
    img_transformed = transform

    # データセット読み込み + 前処理
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                        train=True, download=True, transform=img_transformed)
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                    train=False, download=True, transform=img_transformed)

    # numpyに変換
    train_x = np.array([train_dataset[i][0].cpu().numpy() for i in range(60000)])
    train_y = np.array([train_dataset[i][1] for i in range(60000)])
    test_x = np.array([test_dataset[i][0].cpu().numpy() for i in range(10000)])
    test_y = np.array([test_dataset[i][1] for i in range(10000)])

    # 足し算データセット作成
    group_train_x, group_train_y = make_summation_dataset(60000, 5, "train", train_x, train_y)
    group_test_x, group_test_y = make_summation_dataset(10000, 5, "test", test_x, test_y)

    return group_train_x, group_train_y

if __name__ == "__main__":
    train_loader, val_loader, test_loader = MNIST_load()
    
    for i, (inputs, labels) in enumerate(train_loader):
        imgs = inputs[0].cpu().numpy()

        for img in imgs:
            plt.imshow(img.reshape(28,28))
            plt.show()
            plt.close() 

        print(labels[0])
