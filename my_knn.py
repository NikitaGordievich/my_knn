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
        lens=np.sum(x**2+self.x**2-2*self.x*x,axis=-1)
        inds=np.argsort(lens)
        top_k= self.y[inds[:self.k]]
        vals,counts=np.unique(top_k,return_counts=True)
        return vals[np.argmax(counts)]
    
        

        
if __name__=='__main__':
    from matplotlib import pyplot as plt
    x=np.random.random(size=(100,2))
    mask=x[:,1]>0.5
    y=np.ones_like(100)*mask

    plt.figure(figsize=(10,10))
    x_v=np.random.random(size=(10,2))
    knn=KNN(k=2)
    knn.fit(x,y)
    y_v=knn.predict(x_v)
    print(y_v.shape,y.shape)
    plt.scatter(np.concatenate([x,x_v],0),cmap='virits',c=np.concatenate([y,np.squeeze(y_v)]))
    plt.show()
