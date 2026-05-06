from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

print("Tracking URI:", mlflow.get_tracking_uri())
print("Registered models:", client.search_registered_models())