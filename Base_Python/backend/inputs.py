import numpy as np

# Datas are going to be loaded from here

# y is going to be the target variable
# x1, x2, x3, x4, x5, x6, x7, x8, x9,x10, etc. are going to be the features
# a, b, c, d, e, f, g, h, i, j, etc are going to be the coefficients
# u is going to be the error term
# t is going to be the number of observations

# The equation is going to be:
# Example : y = a + bx1 + cx2 + dx3 + ex4 + fx5 + gx6 + hx7 + ix8 + jx9 + u


# Theory: There is going to be under matricial form, the following equation: Y = X * B + U

# Y is going to be a t x 1 matrix, where n is the number of observations, it contains the target variable
# X is going to be a t * k matrix, where k is the number of features, it contains the features
# B is going to be a k x 1 matrix, where k is the number of features, it contains the coefficients
# U is going to be a t x 1 matrix, where n is the number of observations, it contains the error terms

# X and Y matrices contains data that we know, and we are going to estimate the B matrix that contains the coefficients we must find.

# The goal is to minimize the sum of the squared errors, which is the sum of the squared differences between the observed values of the target variable and the values predicted by the linear approximation.

# The prediction of B matrix is going to be B = (X^T * X)^-1 * X^T * Y


def inputs():
    print("You are going to enter Your data (features and target values), first the features and then the target values, etc.")
    print("The data must be séparated by a comma")

    y_input = input("Enter the values of y (target variable), séparated by a comma ")
    y = np.array(y_input.split(",")).astype(float)
    Y = y.reshape(-1, 1)
    print("loading...")
    t= len(y)
    Y_shape = Y.shape
    print(f"Y matrix shape is {Y_shape}")
    print(Y)

    nbr_features = int(input("Enter the number of features you have "))

    #matrix initialization
    feature_input = input(f"Enter the values of x0, separated by a comma. : ")
    feature = np.array([float(value) for value in feature_input.split(",")], dtype=float)
    features_matrix = feature.reshape(-1, 1)

    for i in range(1, nbr_features):
        feature_input = input(f"Enter the value of x{i} (feature variable), séparated by a comma. ")
        feature = np.array([float(value) for value in feature_input.split(",")], dtype=float)
        features_matrix = np.column_stack((features_matrix, feature))

    ones = np.ones((t, 1))

    # X matrix is going to be the matrix of features with a column of ones
    X = np.column_stack((ones, features_matrix))
    print("loading...")
    print(f"X matric shape is {X.shape}")
    print(X)

    return X, Y, t
    

        

    

