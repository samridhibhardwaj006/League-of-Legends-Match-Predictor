from src.preprocess import prepare_data
from src.model import LogisticRegressionModel
from src.train import train_model
from src.evaluate import evaluate_model
from src.tune import tune_learning_rate
from src.feature_importance import show_feature_importance


DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv"


X_train, X_test, y_train, y_test, feature_names = prepare_data(
    DATA_URL
)

model = LogisticRegressionModel(
    X_train.shape[1]
)

model = train_model(
    model,
    X_train,
    y_train
)

evaluate_model(
    model,
    X_test,
    y_test
)

best_lr, best_acc = tune_learning_rate(
    X_train,
    y_train,
    X_test,
    y_test,
    X_train.shape[1]
)

print(
    f"Best LR: {best_lr}"
)

show_feature_importance(
    model,
    feature_names
)
