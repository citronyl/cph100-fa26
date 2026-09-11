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
        self.feature_names = []
        self.is_fit = False

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """

        values = [float(v) for v in values if v is not None and v != '']
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
            if x is None or x == '':
                 return np.array([0.0])
            return np.array([(float(x) - mean)/std])
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
            bin_idx = np.digitize(x, bin_edges, right=False) - 1
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            one_hot = np.zeros(self.num_bins, dtype=int)
            one_hot[bin_idx] = 1.0

            return one_hot
        return vectorizer
        raise NotImplementedError("Histogram vectorizer not implemented yet")

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector
        """

        # values are not guaranteed to be numerical, so use indexes instead

        cleaned_values = []
        for v in values:
            if str(v).upper() in [".F", ".M"]:
                cleaned_values.append("0")
            else:
                cleaned_values.append(v)

        unique_cats = sorted(list(set(str(v) for v in values)))
        cat_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
        num_cats = len(unique_cats)

        def vectorizer(x):
            one_hot = np.zeros(num_cats, dtype=int)
            val = str(x)
            if val in cat_idx: 
                 one_hot[cat_idx[val]] = 1
            return one_hot

        return vectorizer
        
        raise NotImplementedError("Categorical vectorizer not implemented yet")

    def get_ordinal_vectorizer(self, values):
        """
        :return: function to map ordinal x to integer feature vector
        """
        # replace .F and .M with 0, then sort the unique values and assign an index. 
        cleaned_values = []
        for v in values:
            if str(v).upper() in [".F", ".M"]:
                cleaned_values.append("0")
            else:
                cleaned_values.append(v)

        unique_ords = sorted(list(set(str(v) for v in cleaned_values)))
        ord_idx = {ord: idx for idx, ord in enumerate(unique_ords)}

        def vectorizer(x):
            val = str(x)
            if val in ord_idx:
                return np.array([ord_idx[val]])
            return np.array([0])

        return vectorizer

        raise NotImplementedError("Ordinal vectorizer not implemented yet")
    
    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """

        # raise NotImplementedError("Not implemented yet")
        for feature in self.feature_config.get("numerical", []):
            values = [row[feature] for row in X if feature in row]
            self.feature_transforms[feature] = self.get_numerical_vectorizer(values)
            self.feature_names.append(feature)

        for feature in self.feature_config.get("categorical", []):
                values = [row[feature] for row in X if feature in row]
                self.feature_transforms[feature] = self.get_categorical_vectorizer(values)
                unique_cats = sorted(list(set(str(v) for v in values)))
                for cat in unique_cats:
                    self.feature_names.append(f"{feature}_{cat}")

        for feature in self.feature_config.get("histogram", []):
                values = [row[feature] for row in X if feature in row]
                self.feature_transforms[feature] = self.get_histogram_vectorizer(values)
                for bin_i in range(self.num_bins):
                    self.feature_names.append(f"{feature}_bin_{bin_i}")

        for feature in self.feature_config.get("ordinal", []):
                values = [row[feature] for row in X if feature in row]
                self.feature_transforms[feature] = self.get_ordinal_vectorizer(values)
                self.feature_names.append(feature)
        #self.feature_transforms = { "transform_name": None}

        #updates feature_transforms dict with transform_name: vectorizer function for that feature
        self.is_fit = True


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )

        transformed_data = []

        for row in X:
            row_features = []

            for feature, transform_fn in self.feature_transforms.items():
                raw = row.get(feature, None)
                transformed_val = transform_fn(raw)
                row_features.append(transformed_val)

            concat_row = np.concatenate(row_features)
            transformed_data.append(concat_row)
                
        #raise NotImplementedError("Not implemented yet")

        return np.array(transformed_data)

    def get_feature_names(self):
        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )
        return self.feature_names