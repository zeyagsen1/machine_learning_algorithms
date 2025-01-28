import numpy as np
from matplotlib import pyplot as plt

class Linear_Regression:

    
 def __init__(self, lr = 0.01, n_iters=40):
   self.lr=lr
   self.n_iters=n_iters
   self.aa=None
   self.bb=None
   self.train_x=None
   self.train_y=None
   self.predicted=None
   self.y_test=None
 def gradients(self,n_samples):
    y=np.dot(self.aa,self.train_x)+self.bb
    y1=self.train_y
    print()
    dl_da=np.dot((-2*self.train_x.T/ n_samples),y1-y)
    dl_db=np.dot((-2/n_samples),y1-y)
    return[dl_da,dl_db]

  
 def fit(self,x_train,y_train):
    n_samples,n_features=x_train.shape
    self.aa=np.zeros(n_samples)
    self.bb=np.zeros(n_samples)
    self.train_x=x_train
    self.train_y=y_train
    
    
    
  
    for i in range (self.n_iters):
     
      grad_a,grad_b= self.gradients(n_samples)
      self.aa=self.aa-(self.lr*grad_a)
      self.bb=self.bb-(self.lr*grad_b)
      
      print(f"Epoch {i+1}: Completed ")

    self.aa=np.mean(self.aa)
    self.bb=np.mean(self.bb)
    return [self.aa,self.bb]  
  
  
 def plot(self):    
    
    y_train = self.aa * self.train_x + self.bb
    


    plt.figure(figsize=(8, 5))

    # Plot data points
    plt.scatter(self.train_x, self.train_y)

    # Plot the line
    plt.plot(self.train_x, y_train)

    # Add labels, legend, and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    plt.show()

 def predict(self,x_test,y_test):
    print()
    count=0   
    self.predicted=self.aa*x_test+self.bb
    self.predicted=self.predicted.flatten()
    self.y_test=y_test
    
    
    for x,y in zip(self.predicted,self.y_test): 
     print(f"predicted value={x} real value {y} ")
     count=count+1
     print(count) 
      
 def score(self):
    print()
    error=np.subtract(self.predicted,self.y_test)
    squared_error=np.square(error)
    mean_squared_error=(np.mean(squared_error))
    print(f"mean_squared_error{mean_squared_error}")
    

   
    
     
