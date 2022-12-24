import numpy as np

def logistic(z):
    """
    The logistic function
    Input:
       z   numpy array (any shape)
    Output:
       p   numpy array with same shape as z, where p = logistic(z) entrywise
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    p = np.full(z.shape, 1)
    step1 = np.exp(-1*z)
    step2 = 1 + step1
    p = np.divide(p,step2)
    return p

def cost_function(X, y, theta):
    """
    Compute the cost function for a particular data set and hypothesis (weight vector)
    Inputs:
        X      data matrix (2d numpy array with shape m x n)
        y      label vector (1d numpy array -- length m)
        theta  parameter vector (1d numpy array -- length n)
    Output:
        cost   the value of the cost function (scalar)
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    # for i in range(X.size,0):
    #     #Getting h theta
    #     z = np.dot(np.transpose(theta),X[i])
    #     h_theta = logistic(z)
    #     #First term of cost function
    #     step1 = np.log(h_theta)
    #     first_term = -1 * y[i] * step1
    #     cost = cost + first_term
    #     #Second term of cost function
    #     step1 = np.log(1-h_theta)
    #     second_term = -1 * (1-y[i]) * step1
    #     cost = cost + second_term
    z = np.dot(X,theta)
    h_theta = logistic(z)
    cost = -1*np.sum(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta)) #.sum is a very useful function. Remember it!
    return cost

def gradient_descent( X, y, theta, alpha, iters ):
    """
    Fit a logistic regression model by gradient descent.
    Inputs:
        X          data matrix (2d numpy array with shape m x n)
        y          label vector (1d numpy array -- length m)
        theta      initial parameter vector (1d numpy array -- length n)
        alpha      step size (scalar)
        iters      number of iterations (integer)
    Return (tuple):
        theta      learned parameter vector (1d numpy array -- length n)
        J_history  cost function in iteration (1d numpy array -- length iters)
    """

    # REPLACE CODE BELOW WITH CORRECT CODE
    J_history = [-1] * iters
    for i in range(iters):
        #Compute h_theta
        z = np.dot(X,theta)
        h_theta = logistic(z)
        #Compute gradient
        gradient = np.dot(np.transpose(X),h_theta - y) #This multiplies xji by h_theta(xi) - y
        #Apply gradient to theta
        theta = theta - alpha * gradient
        J_history[i] = cost_function(X,y,theta)
    return theta, J_history

