import re
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn import (
    base,
    compose,
    impute,
    preprocessing,
    decomposition,
    feature_selection,
    impute,
)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class DfTransformer(base.BaseEstimator, base.TransformerMixin):
    """
        Base Class for all transfomers, which outputs the data back to a dataframe object
    """

    def __init__(self, validate_features=True):
        self.columns = []
        self.validate_features = validate_features

    def get_feature_names(self):
        return self.columns

    def match_features(self, X):

        features_missing_from_train = list(set(X.columns) - set(self.columns))
        features_missing_from_test = list(set(self.columns) - set(X.columns))

        missing_features = False

        if len(features_missing_from_test) > 0:
            for feature in features_missing_from_test:
                if X[feature].dtype == object:
                    X[feature] = '0'
                else:
                    X[feature] = 0
                missing_features = True

        if len(features_missing_from_train) > 0:
            for feature in features_missing_from_train:
                del X[feature]
                missing_features = True

        if missing_features:
            warnings.warn(
                'Notice: Certain features in the test data do not match the ones in the training data. This issue has been resolved.',
                UserWarning)

        return X

    def feature_validator(self, X):
        if len(self.columns) > 0:
            if self.validate_features:
                X = self.match_features(X)
        return X


class FeatureTransformer(DfTransformer):
    """
        The Role of this Transformer is to apply transformations to specific features only.
        params:
        1. transformers -> a list of tuples in the form (name,transformer,features), specifying
        transformer objects to be applied to subsets of the data
            a. features -> can be either a list or a Callable make_column_selector to select specific
            columns by name or type
        2. remainder -> {'drop','passthrough'}, if drop then return only defined features, if passthrough
        then the remaining features are concatendated with the output transformed features (to get all of the data).
    """

    def __init__(self,
                 transformers: list,
                 n_jobs=1,
                 sparse_threshold=0.3,
                 transformer_weights=None,
                 verbose=False,
                 remainder="passthrough"):
        self.transformers = transformers
        self.remainder = remainder
        self.n_jobs = n_jobs
        self.sparse_threshold = sparse_threshold
        self.columns = []
        self.column_types = None

        self.feature_transformer = compose.ColumnTransformer(
            transformers=self.transformers,
            remainder=self.remainder,
            n_jobs=self.n_jobs,
            sparse_threshold=self.sparse_threshold
        )

    def fit(self, X, y=None):
        self.get_available_features(X)
        self.feature_transformer.fit(X)
        return self

    def transform(self, X, y=None):
        X_tr = self.feature_transformer.transform(X)
        self.columns = self.feature_transformer.get_feature_names()
        self.adjust_feature_names()

        X_tr = pd.DataFrame(X_tr, index=X.index, columns=self.columns)
        X_tr = self.redefine_feature_types(X_tr)
        return X_tr

    def adjust_feature_names(self):
        for i, col in enumerate(self.columns):
            self.columns[i] = col.split(sep="__")[-1]

    def redefine_feature_types(self, X):

        for col in X:
            X[col] = X[col].convert_dtypes(convert_string=False)
        return X

    def get_params(self, deep=True):
        return self.feature_transformer.get_params(deep=deep)

    def get_available_features(self, X):
        for i, transformer in enumerate(self.transformers):
            transformer_columns = [col for col in transformer[-1] if col in X.columns.tolist()]
            self.transformers[i] = list(self.transformers[i])
            self.transformers[i][-1] = transformer_columns
            self.transformers[i] = tuple(self.transformers[i])

            removed_feature_list = list(set(transformer[-1]) - set(self.transformers[i][-1]))
            if len(removed_feature_list) > 0:
                logging.info(
                    f"The following features: {removed_feature_list} were removed because they were no longer present in the data")


class InputCleaner(DfTransformer):
    def __init__(self,
                 null_pct_threshold: float = 0.9,
                 validate_features: bool = True):

        self.null_pct_threshold = null_pct_threshold
        self.validate_features = validate_features
        self.columns = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_tr = X.copy()

        # remove mostly empty features
        X_tr = self.drop_mostly_empty_features(X_tr)

        # remove uni-valued features
        X_tr = self.drop_only_one_value_features(X_tr)

        # validate that test data and train data have the same features
        X_tr = super().feature_validator(X_tr)
        self.columns = X_tr.columns
        return X_tr

    def drop_mostly_empty_features(self, X):

        not_null_feature_list = X.columns[X.isnull().mean() < self.null_pct_threshold]
        null_feature_list = [feat for feat in X.columns if feat not in not_null_feature_list]

        if len(null_feature_list) > 0:
            warnings.warn(
                f'The following features: {null_feature_list} were removed because they exceeded the null % threshold: {self.null_pct_threshold}')

        return X[not_null_feature_list]

    def drop_only_one_value_features(self, X):

        not_one_value_feature_list = X.columns[X.nunique() != 1]
        one_value_feature_list = [feat for feat in X.columns if feat not in not_one_value_feature_list]

        if len(one_value_feature_list) > 0:
            warnings.warn(
                f'The following features: {one_value_feature_list} were removed because they contain only one value')

        return X[not_one_value_feature_list]

    def get_feature_names(self):
        return self.columns


class Remover(DfTransformer):

    def __init__(self,
                 feature_list: list,
                 validate_features: bool = True):
        self.feature_list = feature_list
        self.validate_features = validate_features
        self.columns = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_tr = X.copy()
        for feature in self.feature_list:
            del X_tr[feature]

        # validate that test data and train data have the same features
        X_tr = super().feature_validator(X_tr)
        self.columns = X_tr.columns
        return X_tr

    def get_feature_names(self):
        return self.columns


class CategoricalEncoder(DfTransformer):
    """
        Categorical feature encoder
    """

    def __init__(self,
                 method: str,
                 drop_first=None,
                 handle_unknown="error",
                 categories="auto"):

        self.columns = []
        self.method = method
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.categories = categories

    def fit(self, X, y=None):
        if self.method == "nominal":
            self.encoder = preprocessing.OneHotEncoder(
                drop=self.drop_first,
                handle_unknown=self.handle_unknown,
                dtype=int,
                sparse=False,
            ).fit(X)
        elif self.method == "ordinal":
            self.encoder = preprocessing.OrdinalEncoder(
                categories=self.categories,
                handle_unknown=self.handle_unknown,
                dtype=int,
            ).fit(X)

        return self

    def transform(self, X, y=None):
        X_tr = X.copy()
        if self.method == 'nominal':
            X_tr = self.nominal_transformation(X_tr, y)
        elif self.method == 'ordinal':
            X_tr = self.ordinal_transformation(X_tr, y)

        return X_tr

    def nominal_transformation(self, X, y=None):
        X_tr = self.encoder.transform(X)
        self.columns = self.encoder.get_feature_names(X.columns)
        X_tr = pd.DataFrame(X_tr, columns=self.columns)
        return X_tr

    def ordinal_transformation(self, X, y=None):
        X_tr = self.encoder.transform(X)
        X_tr = pd.DataFrame(X_tr, index=X.index, columns=X.columns)
        self.columns = X_tr.columns
        return X_tr

    def get_feature_names(self):
        return self.columns


class Imputer(DfTransformer):
    """
        Builds on SimpleImputer, keeps the dataframe structure, available methods:
        1. mean
        2. median
        3. most_frequent
        4. constant
        Note: When method -> “constant”, if fill_value is left to default value, fill_value
        will be 0 for numerical data and “missing_value” for categorical data.

    """

    def __init__(self, method="median", fill_value=None):
        self.method = method
        self.fill_value = fill_value
        self.imputer = None
        self.statistics_ = None
        self.columns = None

    def fit(self, X, y=None):
        self.imputer = impute.SimpleImputer(strategy=self.method, fill_value=self.fill_value)
        self.imputer.fit(X)
        self.statistics_ = pd.Series(self.imputer.statistics_, index=X.columns)
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)

        self.columns = X_imputed_df.columns
        return X_imputed_df