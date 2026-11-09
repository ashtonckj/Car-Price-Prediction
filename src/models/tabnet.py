from common import RANDOM_STATE, evaluate, load_data
from pytorch_tabnet.tab_model import TabNetRegressor

X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()

model = TabNetRegressor(seed=RANDOM_STATE, verbose=10)
model.fit(
    X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=["rmse"],
    max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128,
)
evaluate("TabNet", y_test, model.predict(X_test), inverse_transform_price)