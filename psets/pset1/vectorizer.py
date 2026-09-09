import numpy as np

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.num_bins=5
        self.feature_transforms = {}
        self.is_fit = False

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """

        mean, std = np.mean(values), np.std(values)

        if std == 0:
            std = 1.0

        #raise NotImplementedError("Numerical vectorizer not implemented yet")

        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """

            return (x-mean)/std
            #NotImplementedError("Not implemented")

        return vectorizer

    def get_histogram_vectorizer(self, values):
        """
            :return: function to map continuous x into a one-hot histogram bin vector. 
        """
        _, bin_edges = np.histogram(values, bins=self.num_bins)

        def vectorizer(x):
            """
            :param x: Single raw feature value for a patient
            :return: 1D NumPy array representing one-hot bin assignment
            """
            bin_idx = np.digitize

        #raise NotImplementedError("Histogram vectorizer not implemented yet")

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        raise NotImplementedError("Categorical vectorizer not implemented yet")

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """

        raise NotImplementedError("Not implemented yet")
        self.feature_transforms = { "transform_name": None}
        self.is_fit = True


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )

        transformed_data = []
        raise NotImplementedError("Not implemented yet")

        return np.array(transformed_data)