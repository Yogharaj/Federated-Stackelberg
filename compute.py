import xgboost as xgb
from Client.metrics import compute_metrics_for_datasets

def load_model(model_path):
    """Load the XGBoost model from the specified path."""
    model = xgb.Booster()
    model.load_model(model_path)
    return model

if __name__ == "__main__":
    model_paths = [
        "/Users/yogharajar/Documents/Federated_Stackelberg/Models/aggregated_model_carts.bin",
        "/Users/yogharajar/Documents/Federated_Stackelberg/Models/aggregated_model_catalogue.bin",
        "/Users/yogharajar/Documents/Federated_Stackelberg/Models/aggregated_model_front-end.bin",
        "/Users/yogharajar/Documents/Federated_Stackelberg/Models/aggregated_model_orders.bin",
        "/Users/yogharajar/Documents/Federated_Stackelberg/Models/aggregated_model_payment.bin",
        "/Users/yogharajar/Documents/Federated_Stackelberg/Models/aggregated_model_shipping.bin",
        "/Users/yogharajar/Documents/Federated_Stackelberg/Models/aggregated_model_user.bin"
    ]

    models = [load_model(path) for path in model_paths]

    datasets = ['carts', 'catalogue', 'front-end', 'orders', 'payment', 'shipping', 'user']
    compute_metrics_for_datasets(models, datasets)
