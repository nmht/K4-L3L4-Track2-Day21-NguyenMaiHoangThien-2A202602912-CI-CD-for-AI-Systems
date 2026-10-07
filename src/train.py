import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
os.environ["MLFLOW_ARTIFACT_ROOT"] = "./mlartifacts"
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho GradientBoostingClassifier
        data_path: đường dẫn đến file dữ liệu huấn luyện
        eval_path: đường dẫn đến file dữ liệu đánh giá

    Trả về:
        f1 (float): điểm F1 của lớp dương trên tập holdout
    """

    # 1.6.1: Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 1.6.2: Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 1.6.3: MLflow start_run
    mlflow.set_experiment("Income_Model_Fixed")
    with mlflow.start_run():
        # 1.6.4: Ghi nhận tham số
        mlflow.log_params(params)

        # 1.6.5: Khởi tạo và huấn luyện
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 1.6.6: Tính metrics
        preds = model.predict(X_eval)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)

        # 1.6.7: Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # 1.6.8: Log model
        mlflow.sklearn.log_model(model, "model")

        # 1.6.9: In kết quả
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # 1.6.10: Lưu report.json
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({"f1_score": f1, "accuracy": acc}, f)

        # 1.6.11: Lưu model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # 1.6.12: Trả về f1
    return f1

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
