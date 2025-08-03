from sklearn.base import BaseEstimator, TransformerMixin

class RareClassGroupingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="category", new_col="category_grouped", threshold=0.01):
        self.target_col = target_col
        self.new_col = new_col
        self.threshold = threshold
        self.rare_classes_ = set()

    def fit(self, X, y=None):
        value_counts = X[self.target_col].value_counts(normalize=True)
        self.rare_classes_ = set(value_counts[value_counts < self.threshold].index)
        return self

    def transform(self, X):
        X = X.copy()
        if self.target_col not in X.columns:
            # In inference mode, the category column may not exist — skip
            X[self.new_col] = "unknown"
            return X

        X[self.new_col] = X[self.target_col].apply(
            lambda x: x if x not in self.rare_classes_ else 'unknown'
        )
        return X
