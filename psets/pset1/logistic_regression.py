import numpy as np
import tqdm

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda
        self.train_loss = []
        self.val_loss = []

    def fit(self, X, Y, X_val=None, Y_val=None):
        """
            Train the logistic regression model using stochastic gradient descent.
        """
        num_samples, num_features = X.shape

        # initalize theta and bias as zeros
        self.theta = np.zeros(num_features)
        self.bias = 0.0

        epoch_iter = range(self.num_epochs)
        if self.verbose:
            epoch_iter = tqdm.tqdm(epoch_iter, desc = "Training SGD")

        for _ in epoch_iter:
            indices = np.random.permutation(num_samples)
            X_shuffle = X[indices]
            Y_shuffle = Y[indices]

            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)

                # take random permutation of X and Y to make the batch
                X_batch = X_shuffle[start_idx:end_idx]
                Y_batch = Y_shuffle[start_idx:end_idx]

                # calculate gradients based on gradient func
                gradient_theta, gradient_bias = self.gradient(X_batch, Y_batch)

                # update theta and bias by learning rate and gradient
                self.theta -= self.learning_rate * gradient_theta
                self.bias -= self.learning_rate * gradient_bias

            p_train = self.predict_proba(X)
            loss_train = -np.mean(
                Y * np.log(p_train) + (1-Y) * np.log(1 - p_train)
            )
            self.train_loss.append(float(loss_train))

            if X_val is not None and Y_val is not None:
                p_val = self.predict_proba(X_val)
                loss_v = -np.mean(
                    Y_val * np.log(p_val) + (1-Y_val) * np.log(1 - p_val)
                )
                self.val_loss.append(float(loss_v))
        #raise NotImplementedError("Not implemented yet")

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """
        N = X.shape[0]

        p = self.predict_proba(X)
        error = p - Y

        # get gradient for theta by taking binary cross entropy loss function and finding the derivative through chain rule
        # and adding derivative of L2 regularization function to that
        gradient_theta = (1.0/N) * np.dot(X.T, error) + (self.regularization_lambda * self.theta)
        gradient_bias = (1.0/N) * np.sum(error)

        return gradient_theta, gradient_bias
        raise NotImplementedError("Not implemented yet")

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        # model definition
        z = np.dot(X, self.theta) + self.bias
        p = 1.0 / (1.0 + np.exp(-z))

        #clipping p to avoid probabilities too close to 0 or 1 (improves stability)
        epsilon = 1e-6
        p_clipped = np.clip(p, epsilon, 1.0 - epsilon)
        return p_clipped
        #raise NotImplementedError("Not implemented yet")

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        return (self.predict_proba(X) >= threshold).astype(int)
        #raise NotImplementedError("Not implemented yet")