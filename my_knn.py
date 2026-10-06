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
            self.y=np.concatinate([self.y,y],axis=0)

    def predict(self,x):
        lens=np.sum(x**2+self.x**2-2*self.x*x,axis=-1)
        inds=np.argsort(lens)
        top_k= self.y[inds[:self.k]]
        vals,counts=np.unique(top_k,return_counts=True)
        return vals[np.argmax(counts)]
    
        

        
if __name__=='__main__':
    knn=KNN(k=2)
    print()
