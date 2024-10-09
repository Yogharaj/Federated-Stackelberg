import numpy as np
import pickle
import base64
import xgboost as xgb
from flask import Flask, request, jsonify

app = Flask(__name__)

global_weights = {} 
num_clients = {}  

@app.route("/update_model", methods=["POST"])
def update_model():
    try:
        weights_received = pickle.loads(base64.b64decode(request.json['weights']))
        classname = request.json['classname']
        client_id = request.json['client_id']
        
        print(f"Received weights and history for {classname} from client {client_id}.")

        store_weights_for_averaging(weights_received, classname)

        if num_clients[classname] >= 1: 
            averaged_weights = average_weights(classname)
            save_aggregated_model(averaged_weights, classname)
            num_clients[classname] = 0  

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "failure", "error": str(e)}), 500

def store_weights_for_averaging(weights_received, classname):
    """Store the received weights for future aggregation."""
    global global_weights
    global num_clients

    if classname not in global_weights:
        global_weights[classname] = []
    if classname not in num_clients:
        num_clients[classname] = 0
    
    if 'leader_model' in weights_received:
        leader_model_weights = weights_received['leader_model']

        global_weights[classname].append(leader_model_weights)
        print(f"Weights stored for aggregation for dataset: {classname}.")
    else:
        print("No leader model found in the received weights.")

    num_clients[classname] += 1

def average_weights(classname):
    """Average the weights of all received models for the specific dataset."""
    global global_weights
    
    boosters = [xgb.Booster(model_file=None) for _ in global_weights[classname]]
    for i, weight in enumerate(global_weights[classname]):
        boosters[i].load_model(bytearray(weight))

    averaged_booster = boosters[0]  

    for booster in boosters[1:]:
        averaged_dump = averaged_booster.get_dump()
        booster_dump = booster.get_dump()

    global_weights[classname] = []

    return averaged_booster

def save_aggregated_model(averaged_booster, classname):
    """Save the aggregated model for future use."""
    model_filename = f"/Users/yogharajar/Documents/Federated_Stackelberg/Models/aggregated_model_{classname}.bin"
    averaged_model_weights = averaged_booster.save_raw()
    averaged_booster.save_model(model_filename)
        
    print(f"Averaged model for {classname} saved as {model_filename}.")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
