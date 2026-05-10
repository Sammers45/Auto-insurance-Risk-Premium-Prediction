from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def train_classification_models(X_train, y_train):
    """Train two classification models for claim risk."""
    logistic_model = LogisticRegression(max_iter=200, random_state=42)
    forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

    logistic_model.fit(X_train, y_train)
    forest_model.fit(X_train, y_train)

    return {
        'logistic_regression': logistic_model,
        'random_forest': forest_model,
    }


def train_regression_models(X_train, y_train):
    """Train two regression models for premium prediction."""
    linear_model = LinearRegression()
    forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

    linear_model.fit(X_train, y_train)
    forest_model.fit(X_train, y_train)

    return {
        'linear_regression': linear_model,
        'random_forest': forest_model,
    }
