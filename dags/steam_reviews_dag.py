from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def ingest():
    # imported inside the function so Airflow's DAG parser stays fast
    from src.ingestion.steam import ingest_all
    ingest_all()


def load():
    from src.ingestion.load import load as load_to_db
    load_to_db()


with DAG(
    dag_id="steam_reviews_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule="@daily",
    catchup=False,
    tags=["entertainment", "steam"],
) as dag:
    ingest_task = PythonOperator(task_id="ingest_steam", python_callable=ingest)
    load_task = PythonOperator(task_id="load_postgres", python_callable=load)

    ingest_task >> load_task