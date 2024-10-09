import numpy as np
import requests
import pickle
from preprocessing import load_data, preprocess_data
from model import StackelbergModel
from metrics import evaluate_model
import base64

def send_weights_to_server(weights, history, classname, client_id):
    server_url = "http://localhost:5000/update_model"
    data = {
        'weights': base64.b64encode(pickle.dumps(weights)).decode('utf-8'),  
        'history': history,
        'classname': classname,
        'client_id': client_id
    }
    response = requests.post(server_url, json=data)
    if response.status_code == 200:
        print(f"Client {client_id} successfully sent weights for {classname}.")
    else:
        print(f"Client {client_id} failed to send weights for {classname}.")

def client_train(client_id, classname):
    base_path = "Dataset"
    epochs = 20

    print(f"Client {client_id} processing {classname} dataset...")

    x_train, y_train, x_test, y_test = load_data(classname, base_path)
    x_train_scaled, x_test_scaled = preprocess_data(x_train, x_test)

    print("Completed Processing")
    model = StackelbergModel()
    history = model.train(x_train_scaled, y_train, epochs=epochs)

    weights = model.get_weights()

    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, x_test_scaled, y_test)
    print(f"Metrics for {classname} on client {client_id}:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    send_weights_to_server(weights, history, classname, client_id)

if __name__ == "__main__":
    classname = 'catalogue'  
    client_id = 2 
    client_train(client_id, classname)
