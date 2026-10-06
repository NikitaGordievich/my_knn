
from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt

def load():
    data=load_iris()
    x=data['data']
    y=data['target']
    ind=np.arange(x.shape[0])
    x=x[ind]
    y=y[ind]
    x_train=x[:120]
    y_train=y[:120]
    x_val=x[120:]
    y_val=y[120:]
    return x_train,y_train,x_val,y_val

def visualis(data):
    x=np.arange(0,data.shape[0])
    plt.figure(figsize=(8,10))
    for i in range(data.shape[-1]):
        plt.subplot(data.shape[-1],1,i+1)
        y=data[:,i]
        plt.scatter(x,y)
    plt.show()

if __name__=='__main__':
    d=load()
    visualis(d[0])
