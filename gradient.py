
########################################################### Without the bias term #############################################################

import numpy as np

# This function finds the new gradient at each step
def step_gradient(points, learning_rate, m):
    M = len(points)
    num_features = points.shape[1] - 1  # Number of features (excluding the target variable)
    m_slope = np.zeros(num_features + 1)  # Initialize slopes for all features and the bias term
    
    # Add a column of ones to the feature matrix for the bias term
    X = np.hstack((points[:, :-1], np.ones((M, 1))))  # Adding bias term column
    y = points[:, -1]  # Target values

    # Compute gradients
    for i in range(M):
        xi = X[i]  # Features including bias term
        yi = y[i]   # Target value
        prediction = np.dot(xi, m)  # Compute prediction with bias term included
        error = yi - prediction
        
        # Update gradients for all features including the bias term
        m_slope += (-2/M) * error * xi
    
    # Update weights
    new_m = m - learning_rate * m_slope
    
    return new_m

# The Gradient Descent Function
def gd(points, learning_rate, num_iterations):
    num_features = points.shape[1] - 1  # Number of features (excluding the target variable)
    m = np.zeros(num_features + 1)  # Initialize weights including the bias term
    
    for i in range(num_iterations):
        m = step_gradient(points, learning_rate, m)
        print(i, " Cost: ", cost(points, m))
    
    return m

# This function finds the new cost after each optimisation.
def cost(points, m):
    total_cost = 0
    M = len(points)
    num_features = points.shape[1] - 1  # Number of features (excluding the target variable)
    
    # Add a column of ones to the feature matrix for the bias term
    X = np.hstack((points[:, :-1], np.ones((M, 1))))  # Adding bias term column
    y = points[:, -1]  # Target values
    
    # Compute total cost
    for i in range(M):
        xi = X[i]  # Features including bias term
        yi = y[i]   # Target value
        prediction = np.dot(xi, m)  # Compute prediction with bias term included
        total_cost += (1/M) * (yi - prediction)**2
    
    return total_cost

def run():
    # Example of loading training data, which should be in the form of a numpy array
    # Each row should be of the form [x1, x2, ..., xn, y] where x1, x2, ..., xn are features and y is the target variable
    training_data = np.array([
        [1, 2, 3, 4],  # Example row with 3 features and 1 target value
        [2, 3, 4, 5],
        [3, 4, 5, 6]
        # Add more rows as needed
    ])
    
    learning_rate = 0.0001
    num_iterations = 100
    m = gd(training_data, learning_rate, num_iterations)
    
    # Split weights into feature weights and bias term
    feature_weights = m[:-1]
    bias = m[-1]
    
    print("Final weights (m):", feature_weights)
    print("Final bias (c):", bias)
    return feature_weights, bias

feature_weights, bias = run()














########################################################### With the bias term #############################################################
import numpy as np

# This function finds the new gradient at each step
def step_gradient(points, learning_rate, m, c):
    num_features = points.shape[1] - 1  # Number of features (excluding the target variable)
    m_slope = np.zeros(num_features)    # Initialize slopes for each feature
    c_slope = 0                         # Initialize slope for the bias term
    M = len(points)
    
    # Compute gradients
    for i in range(M):
        x = points[i, :-1]  # Features
        y = points[i, -1]   # Target value
        prediction = 0
        for j in range(num_features):
            prediction += m[j] * x[j]  # Compute prediction for each feature
        prediction += c  # Add bias term
        error = y - prediction
        
        for j in range(num_features):
            m_slope[j] += (-2/M) * error * x[j]  # Gradient for each feature
        c_slope += (-2/M) * error  # Gradient for the bias term
    
    # Update weights
    new_m = m - learning_rate * m_slope
    new_c = c - learning_rate * c_slope
    
    return new_m, new_c

# The Gradient Descent Function
def gd(points, learning_rate, num_iterations):
    num_features = points.shape[1] - 1  # Number of features (excluding the target variable)
    m = np.zeros(num_features)  # Initialize weights (m) for all features
    c = 0  # Initialize bias (c)
    
    for i in range(num_iterations):
        m, c = step_gradient(points, learning_rate, m, c)
        print(i, " Cost: ", cost(points, m, c))
    
    return m, c

# This function finds the new cost after each optimisation.
def cost(points, m, c):
    total_cost = 0
    M = len(points)
    num_features = points.shape[1] - 1  # Number of features (excluding the target variable)
    
    # Compute total cost
    for i in range(M):
        x = points[i, :-1]  # Features
        y = points[i, -1]   # Target value
        prediction = 0
        for j in range(num_features):
            prediction += m[j] * x[j]  # Compute prediction for each feature
        prediction += c  # Add bias term
        total_cost += (1/M) * (y - prediction)**2
    
    return total_cost

def run():
    # Example of loading training data, which should be in the form of a numpy array
    # Each row should be of the form [x1, x2, ..., xn, y] where x1, x2, ..., xn are features and y is the target variable
    training_data = np.array([
        [1, 2, 3, 4],  # Example row with 3 features and 1 target value
        [2, 3, 4, 5],
        [3, 4, 5, 6]
        # Add more rows as needed
    ])
    
    learning_rate = 0.0001
    num_iterations = 100
    m, c = gd(training_data, learning_rate, num_iterations)
    
    print("Final weights (m):", m)
    print("Final bias (c):", c)
    return m, c

m, c = run()
