import load_dataset
import numpy as np
import my_knn

print('start')
model=my_knn.KNN(k=2)
print('knn cteate')
x_train,y_train,x_v,y_v=load_dataset.load()
print('data loaded')
model.fit(x_train,y_train)
print('training model end')
y_pred=model.predict(x_v)
accur=np.sum(y_v==y_pred)/y_v.shape[0]
print(f'accuratly: {accur:.4f}')
model.visual_dots(x_v,y_v)


