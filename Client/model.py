import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class StackelbergModel:
    def __init__(self):
        self.leader_model = xgb.XGBClassifier(eval_metric='logloss')
        self.follower_models = []
    
    def train(self, X_train, y_train, epochs=10):
        history = {'loss': []}
        for epoch in range(epochs):
            self.leader_model.fit(X_train, y_train)
            loss = 1 - accuracy_score(y_train, self.leader_model.predict(X_train))
            history['loss'].append(loss)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
        return history
    
    def get_weights(self):
        """Extract model weights as a binary blob (for aggregation)."""
        booster_bytes = self.leader_model.get_booster().save_raw()  
        return {'leader_model': booster_bytes}
    
    def set_weights(self, weights):
        """Set model weights (for aggregation)."""
        booster = xgb.Booster()
        booster.load_model(weights['leader_model'])
        self.leader_model._Booster = booster  
    
    def predict(self, X_test):
        return self.leader_model.predict(X_test)

    def add_follower_model(self, follower_model):
        self.follower_models.append(follower_model)

    def evaluate_followers(self, X_test, y_test):
        results = {}
        leader_predictions = self.leader_model.predict(X_test)
        for follower in self.follower_models:
            predictions = follower.predict(X_test, leader_predictions)
            results[follower.__class__.__name__] = predictions
        return results


class FollowerModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test, leader_predictions):
        return self.model.predict(X_test)
