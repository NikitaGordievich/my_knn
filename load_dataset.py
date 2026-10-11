
from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt

def load():
    np.random.seed(4)
    data=load_iris()
    x=data['data']
    y=data['target']
    ind=np.arange(x.shape[0])
    np.random.shuffle(ind)
    x=x[ind]
    y=y[ind]
    x_train=x[:120]
    x_train,mean,std=preprocessing(x_train)
    y_train=y[:120]
    x_val=x[120:]
    x_val,mean,std=preprocessing(x_val,mean,std)
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

def analitic_corr(data):
    corr=np.corrcoef(data,rowvar=False)
    cor=plt.imshow(corr,cmap='coolwarm',vmin=-1, vmax=1)
    plt.colorbar(cor,shrink=0.8)
    for s in range(corr.shape[0]):
        for i in range(corr.shape[1]):
            plt.text(i,s,f"{corr[s,i]:.2f}",color='black',fontsize=12,ha='center',va='center')
    plt.show()

def preprocessing(data,mean=None,std=None):
    #data=np.delete(data,[2,],axis=-1)
    if mean is None:
        mean=np.mean(data,axis=0)
    if std is None:
        std=np.std(data,axis=0)
    data=(data-mean)/std
    return data,mean,std

if __name__=='__main__':
    d=load()
    visualis(d[0])
    analitic_corr(d[0])
