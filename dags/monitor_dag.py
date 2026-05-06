"""
monitor_dag.py — Daily drift monitoring pipeline

Scheduled daily. For each new window:
1. Compute PSI + KS drift statistics vs reference distribution
2. Compute model performance metrics (AUPRC, Recall@FPR, score dist PSI)
3. Persist all metrics to drift_store.db
4. If drift exceeds thresholds → trigger retraining DAG
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logger = logging.getLogger(__name__)

WINDOWS_PATH = Path("data/windows")
REFERENCE_PATH = WINDOWS_PATH / "reference.parquet"

default_args = {"owner": "ml-team", "retries": 1, "retry_delay": timedelta(minutes=5)}


def get_current_window(**context) -> int:
    try:
        window_id = int(context["dag_run"].conf.get("window_id", 1))
    except (TypeError, ValueError):
        day_offset = (context["logical_date"] - datetime(2024, 1, 1)).days
        window_id = (day_offset % 6) + 1
    context["task_instance"].xcom_push(key="window_id", value=window_id)
    return window_id


def compute_data_drift(**context):
    from drift import detect_drift, init_db
    init_db()
    ti = context["task_instance"]
    window_id = ti.xcom_pull(task_ids="get_current_window", key="window_id")
    current_df = pd.read_parquet(WINDOWS_PATH / f"window_{window_id}" / "data.parquet")
    reference_df = pd.read_parquet(REFERENCE_PATH)
    results = detect_drift(window_id, current_df, reference_df)
    ti.xcom_push(key="drift_detected", value=results["_summary"]["drift_detected"])
    ti.xcom_push(key="window_id", value=window_id)


def compute_model_performance(**context):
    import mlflow.sklearn
    import features as feat_module
    from drift import check_performance_drift, compute_score_distribution_psi, write_performance_metrics, load_config, init_db
    init_db()
    ti = context["task_instance"]
    window_id = ti.xcom_pull(task_ids="get_current_window", key="window_id")
    current_df = pd.read_parquet(WINDOWS_PATH / f"window_{window_id}" / "data.parquet")
    reference_df = pd.read_parquet(REFERENCE_PATH)
    try:
        model = mlflow.sklearn.load_model("models:/fraud-detector/Production")
        pipeline = feat_module.load_pipeline()
    except Exception as e:
        logger.warning(f"No champion model: {e}")
        ti.xcom_push(key="performance_drift", value=False); return
    X_curr, y_curr = feat_module.transform(current_df, pipeline)
    X_ref, _ = feat_module.transform(reference_df, pipeline)
    curr_scores = model.predict_proba(X_curr)[:, 1]
    ref_scores = model.predict_proba(X_ref)[:, 1]
    if y_curr.sum() == 0:
        score_psi = compute_score_distribution_psi(ref_scores, curr_scores)
        config = load_config()
        perf_drift = score_psi > config["drift"]["thresholds"]["score_psi"]
        write_performance_metrics(window_id, {"score_dist_psi": score_psi})
    else:
        metrics = check_performance_drift(window_id, y_curr, curr_scores, ref_scores)
        config = load_config()
        perf_drift = (metrics["auprc"] < 0.70 or
                      metrics["score_dist_psi"] > config["drift"]["thresholds"]["score_psi"])
    ti.xcom_push(key="performance_drift", value=perf_drift)


def should_retrain(**context) -> str:
    ti = context["task_instance"]
    data_drift = ti.xcom_pull(task_ids="compute_data_drift", key="drift_detected") or False
    perf_drift = ti.xcom_pull(task_ids="compute_model_performance", key="performance_drift") or False
    if data_drift or perf_drift:
        reason = "+".join(filter(None, ["data_drift" if data_drift else "", "performance_drift" if perf_drift else ""]))
        ti.xcom_push(key="trigger_reason", value=reason)
        return "trigger_retraining"
    return "no_retraining_needed"


with DAG(dag_id="monitor_drift", default_args=default_args,
         description="Daily drift monitoring", schedule="0 6 * * *",
         start_date=datetime(2024, 1, 1), catchup=False, max_active_runs=1,
         tags=["monitoring", "drift"]) as dag:

    t_window  = PythonOperator(task_id="get_current_window",      python_callable=get_current_window)
    t_data    = PythonOperator(task_id="compute_data_drift",       python_callable=compute_data_drift)
    t_perf    = PythonOperator(task_id="compute_model_performance",python_callable=compute_model_performance)
    t_branch  = BranchPythonOperator(task_id="should_retrain",     python_callable=should_retrain)
    t_trigger = TriggerDagRunOperator(task_id="trigger_retraining", trigger_dag_id="retrain_model",
                    conf={"trigger_reason": "{{ task_instance.xcom_pull(task_ids='should_retrain', key='trigger_reason') }}",
                          "window_id": "{{ task_instance.xcom_pull(task_ids='get_current_window', key='window_id') }}"})
    t_skip    = EmptyOperator(task_id="no_retraining_needed")

    t_window >> [t_data, t_perf] >> t_branch >> [t_trigger, t_skip]