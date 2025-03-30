import numpy as np
import random
# import matplotlib.pyplot as plt

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU activation function."""
    return (x > 0).astype(float)

def softmax(z):
    """
    Compute the softmax of the input array z.
    Args:
        z: Input array (NumPy array)
    Returns:
        Softmax probabilities
    """
    # Ensure z is a NumPy array
    z = np.array(z)

    # Compute softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class FoodNeuralNetwork(object):
    def __init__(self, num_features=332, num_hidden=100, num_classes=128, lambda_reg=0.01):
        """
        Initialize the weights and biases of this two-layer MLP.
        """
        # Adding L2 regularization
        self.l2_lambda = lambda_reg

        # information about the model architecture
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        # weights and biases for the first layer of the MLP
        self.W1 = np.zeros([num_hidden, num_features])
        self.b1 = np.zeros([num_hidden])

        # weights and biases for the second layer of the MLP
        self.W2 = np.zeros([num_classes, num_hidden])
        self.b2 = np.zeros([num_classes])

        # initialize the weights and biases
        self.initializeParams()

        # set all values of intermediate variables (to be used in the
        # forward/backward passes) to None
        self.cleanup()


    def initializeParams(self):
        """
        Initialize the weights and biases of this two-layer MLP to be random.
        This random initialization is necessary to break the symmetry in the
        gradient descent update for our hidden weights and biases. If all our
        weights were initialized to the same value, then their gradients will
        all be the same!
        """
        self.W1 = np.random.normal(0, 2/self.num_features, self.W1.shape)
        self.b1 = np.random.normal(0, 2/self.num_features, self.b1.shape)
        self.W2 = np.random.normal(0, 2/self.num_hidden, self.W2.shape)
        self.b2 = np.random.normal(0, 2/self.num_hidden, self.b2.shape)

    def forward(self, X):
        """
        Compute the forward pass to produce prediction for inputs.

        Parameters:
            `X` - A numpy array of shape (N, self.num_features)

        Returns: A numpy array of predictions of shape (N, self.num_classes)
        """
        return do_forward_pass(self, X) # To be implemented below

    def backward(self, ts):
        """
        Compute the backward pass, given the ground-truth, one-hot targets.

        You may assume that the `forward()` method has been called for the
        corresponding input `X`, so that the quantities computed in the
        `forward()` method is accessible.

        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        return do_backward_pass(self, ts) # To be implemented below

    def loss(self, ts):
        """
        Compute the average cross-entropy loss with L2 regularization
        
        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        # Ensure no zero or negative values in log calculation
        # Add a small epsilon to prevent log(0)
        epsilon = 1e-15
        safe_y = np.clip(self.y, epsilon, 1 - epsilon)
        
        # Cross-entropy loss with safe logarithm
        cross_entropy_loss = np.sum(-ts * np.log(safe_y)) / ts.shape[0]
        
        # L2 regularization term
        l2_loss = (
            np.sum(self.W1 ** 2) + 
            np.sum(self.W2 ** 2)
        )
        
        # Combine cross-entropy loss with L2 regularization
        total_loss = cross_entropy_loss + 0.5 * self.l2_lambda * l2_loss
        
        return total_loss

    def update(self, alpha):
        """
        Compute the gradient descent update for the parameters of this model.

        Parameters:
            `alpha` - A number representing the learning rate
        """
        self.W1 = self.W1 - alpha * (self.W1_bar + self.l2_lambda * self.W1)
        self.b1 = self.b1 - alpha * self.b1_bar
        self.W2 = self.W2 - alpha * (self.W2_bar + self.l2_lambda * self.W2)
        self.b2 = self.b2 - alpha * self.b2_bar

    def cleanup(self):
        """
        Erase the values of the variables that we use in our computation.
        """
        self.N = None # Number of data points in the batch
        self.X = None # The input matrix
        self.m = None # Pre-activation value of the hidden state, should have shape
        self.h = None # Post-RELU value of the hidden state
        self.z = None # The logit scores (pre-activation output values)
        self.y = None # Class probabilities (post-activation)
        # To be filled in during the backward pass
        self.z_bar = None # The error signal for self.z2
        self.W2_bar = None # The error signal for self.W2
        self.b2_bar = None # The error signal for self.b2
        self.h_bar = None  # The error signal for self.h
        self.m_bar = None # The error signal for self.z1
        self.W1_bar = None # The error signal for self.W1
        self.b1_bar = None # The error signal for self.b1


    def predict(self, X):
        """
        Make class predictions for input X (outputs class indices).
        
        Parameters:
            X : numpy array of shape (N, num_features)
            
        Returns:
            numpy array of shape (N,) containing class indices (0, 1, or 2)
        """
        # Get class probabilities using existing forward pass
        proba = self.forward(X)  
        # Return index of maximum probability
        return np.argmax(proba, axis=1)  


def do_forward_pass(model, X):
    """
    Compute the forward pass to produce prediction for inputs.

    This function also keeps some of the intermediate values in
    the neural network computation, to make computing gradients easier.

    For the ReLU activation, you may find the function `np.maximum` helpful

    Parameters:
        `model` - An instance of the class MLPModel
        `X` - A numpy array of shape (N, model.num_features)

    Returns: A numpy array of predictions of shape (N, model.num_classes)
    """
    model.N = X.shape[0]
    model.X = X
    model.m = X.dot(model.W1.T) + model.b1 # the hidden state value (pre-activation)
    model.h = np.maximum(0, model.m) # the hidden state value (post ReLU activation)
    model.z = model.h.dot(model.W2.T) + model.b2  # the logit scores (pre-activation)
    model.y = softmax(model.z) # the class probabilities (post-activation)
    return model.y

def do_backward_pass(model, ts):
    """
    Compute the backward pass, given the ground-truth, one-hot targets.

    You may assume that `model.forward()` has been called for the
    corresponding input `X`, so that the quantities computed in the
    `forward()` method is accessible.

    The member variables you store here will be used in the `update()`
    method. Check that the shapes match what you wrote in Part 2.

    Parameters:
        `model` - An instance of the class MLPModel
        `ts` - A numpy array of shape (N, model.num_classes)
    """
    model.z_bar = (model.y - ts) / model.N
    model.W2_bar = model.z_bar.T.dot(model.h)
    model.b2_bar = np.sum(model.z_bar, axis=0)
    model.h_bar = model.z_bar.dot(model.W2)
    model.m_bar = model.h_bar * (model.m > 0)
    model.W1_bar = model.m_bar.T.dot(model.X)
    model.b1_bar = np.sum(model.m_bar, axis=0)