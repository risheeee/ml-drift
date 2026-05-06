"""
retrain_dag.py — Triggered retraining pipeline (never scheduled)

Triggered by monitor_dag when drift is detected.
1. Train challenger on windows 1..N (point-in-time correct)
2. Run all validation gates
3. If gates pass → promote to Production + hot-reload serving API
4. If gates fail → alert and keep champion
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logger = logging.getLogger(__name__)

default_args = {"owner": "ml-team", "retries": 0}


def run_training(**context):
    from train import train
    conf = context["dag_run"].conf or {}
    run_id = train(up_to_window=int(conf.get("window_id", 3)),
                   run_hpo=True, n_trials=30,
                   trigger_reason=conf.get("trigger_reason", "manual"))
    context["task_instance"].xcom_push(key="challenger_run_id", value=run_id)


def run_validation_gates(**context):
    from gates import validate_and_promote
    ti = context["task_instance"]
    run_id = ti.xcom_pull(task_ids="run_training", key="challenger_run_id")
    report = validate_and_promote(run_id)
    ti.xcom_push(key="gates_passed", value=report.overall_passed)
    ti.xcom_push(key="challenger_run_id", value=run_id)


def should_promote(**context) -> str:
    passed = context["task_instance"].xcom_pull(task_ids="run_validation_gates", key="gates_passed")
    return "promotion_succeeded" if passed else "promotion_rejected"


def hot_reload_serving_api(**context):
    import yaml
    with open(Path(__file__).parent.parent / "configs/config.yaml") as f:
        config = yaml.safe_load(f)
    try:
        resp = requests.post(config["serving"]["reload_endpoint"], timeout=10)
        resp.raise_for_status()
        logger.info(f"API reloaded: {resp.json()}")
    except Exception as e:
        logger.error(f"Hot-reload failed: {e}. New model is in registry; picks up on next restart.")


def log_rejection(**context):
    run_id = context["task_instance"].xcom_pull(task_ids="run_validation_gates", key="challenger_run_id")
    logger.warning(f"Challenger {run_id} rejected by validation gates. Champion remains in production.")


with DAG(dag_id="retrain_model", default_args=default_args,
         description="Triggered retraining with validation gates",
         schedule=None, start_date=datetime(2024, 1, 1),
         catchup=False, max_active_runs=1,
         tags=["training", "retraining"]) as dag:

    t_train    = PythonOperator(task_id="run_training",           python_callable=run_training,
                                execution_timeout=timedelta(hours=2))
    t_gates    = PythonOperator(task_id="run_validation_gates",   python_callable=run_validation_gates)
    t_branch   = BranchPythonOperator(task_id="should_promote",   python_callable=should_promote)
    t_promoted = EmptyOperator(task_id="promotion_succeeded")
    t_rejected = EmptyOperator(task_id="promotion_rejected")
    t_reload   = PythonOperator(task_id="hot_reload_serving_api", python_callable=hot_reload_serving_api)
    t_log_rej  = PythonOperator(task_id="log_rejection",          python_callable=log_rejection)

    t_train >> t_gates >> t_branch
    t_branch >> t_promoted >> t_reload
    t_branch >> t_rejected >> t_log_rej