import matplotlib.pyplot as plt
from matplotlib.ticker import *
from matplotlib import rc

def setting(size, dpi=100.0):

    # サイズ指定（figsizeには出力サイズ/dpiの値を使用）
    figsize = (size[0]/dpi, size[1]/dpi)
    print(figsize)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot(111)

    # グラフの軸・目盛りを全て非表示
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params('x', length=0, which='major')
    ax.tick_params('y', length=0, which='major')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

if __name__ == '__main__':

    rc('text', usetex=True)

    setting(size=(100,100))

    x, y = 0.5, 0.5
    text = "$1 + 2 \times 3  $"
    fontsize = 10

    plt.text(x, y, text, horizontalalignment='center', fontsize=fontsize)

    plt.savefig("test.png")