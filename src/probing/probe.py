import numpy as np
import lightgbm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,classification_report
from src.extraction.activations import ActivationExtractor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
class HallucinationProbe:
    def __init__(self,layer:int=22):
        self.trained_models = {}
        self.extractor = ActivationExtractor(layer=layer)
        self.scaler=StandardScaler()
        self.models={
            "logistic regression":LogisticRegression(max_iter=1000, class_weight="balanced"),
            "SVM":SVC(kernel="rbf", probability=True, class_weight="balanced"),
            "MLP": MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000, early_stopping=True),
            "lightgbm":lightgbm.LGBMClassifier(n_estimators=300, learning_rate=0.05, is_unbalance=True, verbose=-1)
        }
        self.results={}
    def fit(self,X,Y,test_size=0.2,random_state=42):
        X_train,X_test,Y_train,Y_test = train_test_split(
            X,Y,test_size=test_size,random_state=random_state)
        X_train=self.scaler.fit_transform(X_train)
        X_test=self.scaler.transform(X_test)
        for name,model in self.models.items():
            print(f"\n Training {name}...")
            model.fit(X_train,Y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            f1=f1_score(Y_test,y_pred)
            roc = roc_auc_score(Y_test, y_prob[:, 1])  # ← correct, probability of class 1
            self.trained_models[name] = model
            self.results[name] = {
                "f1_score": f1,
                "roc_auc": roc
            }
    def predict(self,X):
        X_scaled=self.scaler.transform(X)
        predictions={}
        for name,model in self.trained_models.items():
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]
            predictions[name] = {
                "prediction": int(pred),
                "hallucination_prob": float(prob)
            }
        return predictions
    def save(self, path="models/"):
        import os
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        for name, model in self.trained_models.items():
            joblib.dump(model, os.path.join(path, f"{name}.pkl"))

    def load(self, path="models/"):
        import os
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        for name in self.models.keys():
            self.trained_models[name] = joblib.load(os.path.join(path, f"{name}.pkl"))
