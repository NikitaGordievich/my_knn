import numpy as np

class KNN():
    def __init__(self,k=1):
        self.k=k
        self.x=None
        self.y=None

    def fit(self,x,y):

        if self.x is None:
            self.x=x
            self.y=y
        else:
            self.x=np.concatinate([self.x,x],axis=0)
            self.y=np.concatenate([self.y,y],axis=0)

    def predict(self,x):
        lens=np.sqrt(np.sum(x**2,axis=-1,keepdims=True)+np.sum(self.x**2,axis=-1)-2*x@self.x.T)
        inds=np.argsort(lens,axis=-1)
        top_k= self.y[inds[:,:self.k]]
        vals,counts=np.unique(top_k,return_counts=True,axis=-1)
        return vals[:,np.argmax(counts,axis=-1)]
    
        

        
if __name__=='__main__':
    from matplotlib import pyplot as plt
    x=np.random.random(size=(1000,2))
    mask=x[:,1]>0.5
    y=np.ones_like(1000)*mask

    plt.figure(figsize=(100,10))
    x_v=np.random.random(size=(100,2))
    knn=KNN(k=1)
    knn.fit(x,y)
    y_v=knn.predict(x_v)
    print(y_v.shape,y.shape)
    data=np.concatenate([x,x_v],0)
    print(data.shape)
    plt.scatter(x[:,0],x[:,1],cmap='viridis',c=y)
    plt.scatter(x_v[:,0],x_v[:,1],cmap='cool',c=y_v,linewidths=1,edgecolors='black')
    plt.show()
