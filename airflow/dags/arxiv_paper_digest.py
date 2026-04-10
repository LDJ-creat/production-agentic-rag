from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from arxiv_ingestion.digest import generate_daily_paper_digest, publish_daily_paper_digest


default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 8),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "catchup": False,
}


dag = DAG(
    "arxiv_paper_digest",
    default_args=default_args,
    description="Daily arXiv paper digest: score recent papers, generate markdown summary, and push to Telegram",
    schedule="15 7 * * *",
    max_active_runs=1,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["arxiv", "digest", "report", "telegram", "papers"],
)


generate_digest_task = PythonOperator(
    task_id="generate_daily_paper_digest",
    python_callable=generate_daily_paper_digest,
    dag=dag,
)


publish_digest_task = PythonOperator(
    task_id="publish_daily_paper_digest",
    python_callable=publish_daily_paper_digest,
    dag=dag,
)


generate_digest_task >> publish_digest_task