from lineer_regression import Linear_Regression
from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split

x,y=make_regression(n_samples=300,n_features=1,n_targets=1,noise=10)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=False)
print(len(x_train))
model=Linear_Regression()
model.n_iters=70
model.lr=0.01
model.fit(x_train,y_train)
model.predict(x_test,y_test)
model.plot()
model.score()


 
 
