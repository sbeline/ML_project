import numpy as np
import matplotlib.pyplot as plt

def cost_function(x,y,theta):
	return np.sum(np.power(x.dot(theta)-y, 2)) *1/(2*y.size)

def gradientDescent(x,y,alpha,iteration,theta):
	x_transpos = np.transpose(x)
	for i in range(0,iteration):
		gradient = np.dot(x_transpos, np.dot(x, theta) - y) / y.size
		theta = theta - alpha * gradient
	return theta

data = np.loadtxt("ex1data1.txt", dtype=float, delimiter=',')
X = data[:,:1]
y = data[:,1:]

X = np.concatenate((np.ones((X.size,1)),X),axis=1)

theta = np.zeros((2,1))
cost_function(X,y,theta)

iteration = 1500
alpha=0.3
theta = gradientDescent(X,y,alpha,iteration,theta)
