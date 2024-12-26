import numpy as np

# To estimate we will suppose U = matrix of zeros (no error terms)

# The prediction of B matrix is going to be B = (X^T * X)^-1 * X^T * Y

def estimation(X, Y):
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    return B

def var_cal(X, Y, B):
    t = X.shape[0]
    k = X.shape[1]

    #Estimation of U
    U = Y - X @ B
    sigma2 = (U.T @ U) / (t - k)
    std = sigma2**0.5
    var = sigma2 * np.linalg.inv(X.T @ X)
    std_B = np.sqrt(np.diagonal(var))
    return std, std_B

def Final_answer(X,Y):
    print(f"Estimated coefficients are: {estimation(X, Y)}")
    print(f"Standart deviation (difference between estimated and real values) is {var_cal(X,Y,estimation(X,Y))[0]}")
    print(f"Standart deviation of the coefficients are: {var_cal(X,Y,estimation(X,Y))[1]}")