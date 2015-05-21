from sklearn import datasets
from sklearn.cross_validation import cross_val_predict 
# cross_val_predict(estimator,X,y,cv,n_jobs) default for cv is 3; n_jobs: optional, # of CPU use to do the computation, -1 = all CPUs  
# cross_val_predict return predicted values (array)
from sklearn import linear_model
import matplotlib.pyplot as plt 
import pylab

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target 

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(lr, boston.data, y, cv = 10)

fig, ax = plt.subplots()
ax.scatter(y,predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
pylab.show()  # use pylab.show() in order to show figure not fig.show() here 
