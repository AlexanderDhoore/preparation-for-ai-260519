from __future__ import annotations

import datetime
import os
import re
import time
from pathlib import Path

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from kubernetes import client, config
from kubernetes.client import ApiException

TAXI_SCRIPT_PATH = Path(
    os.getenv(
        "CHAPTER06_TAXI_SCRIPT_PATH",
        "/opt/airflow/jobs/taxi_airflow_job.py",
    )
)

DEFAULT_CONFIG = {
    "participant": "preparation-for-ai-dev",
    "namespace": "preparation-for-ai-dev",
    "shared_bucket": "preparation-for-ai-shared",
    "output_bucket": "preparation-for-ai-dev",
    "input_month": "2024-01",
    "minimum_rows": 100_000,
    "s3_endpoint": "http://s3.mechatronics.lan",
}

JOB_IMAGE = "python:3.12-slim"
JOB_WORK_DIR = "/opt/course"
JOB_DEPENDENCIES = [
    "boto3==1.43.6",
    "duckdb==1.5.2",
    "pandas==2.3.3",
    "polars==1.40.1",
    "pyarrow==22.0.0",
]


def merged_config(context) -> dict[str, object]:
    run_config = dict(DEFAULT_CONFIG)
    run_config.update(context["dag_run"].conf or {})
    return run_config


def dns_name(value: str, limit: int = 63) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    cleaned = re.sub(r"-+", "-", cleaned)
    return (cleaned or "run")[:limit].rstrip("-")


def run_name(context) -> str:
    return dns_name(f"chapter06-taxi-{context['run_id']}", 52)


def output_prefix(context) -> str:
    return f"outputs/chapter06/taxi-python/run_id={dns_name(context['run_id'], 80)}/"


def list_job_pods(core_api, namespace: str, job_name: str):
    selectors = [
        f"batch.kubernetes.io/job-name={job_name}",
        f"job-name={job_name}",
    ]
    for selector in selectors:
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=selector,
        ).items
        if pods:
            return pods
    return []


def print_job_pod_logs(core_api, namespace: str, job_name: str) -> None:
    pods = list_job_pods(core_api, namespace, job_name)
    if not pods:
        print(f"No Kubernetes pods found for Job {namespace}/{job_name}.")
        return

    for pod in pods:
        pod_name = pod.metadata.name
        print(f"--- Kubernetes pod logs: {namespace}/{pod_name} ---")
        for container in pod.spec.containers:
            print(f"--- container: {container.name} ---")
            try:
                logs = core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    container=container.name,
                )
            except ApiException as exc:
                print(f"Could not read logs for {pod_name}/{container.name}: {exc}")
                continue
            print(logs or "(no logs)")


def validate_run_config(**context) -> None:
    run_config = merged_config(context)
    required = [
        "participant",
        "namespace",
        "shared_bucket",
        "output_bucket",
        "input_month",
        "minimum_rows",
        "s3_endpoint",
    ]
    missing = [name for name in required if not run_config.get(name)]
    if missing:
        raise ValueError(f"Missing run configuration values: {missing}")

    minimum_rows = int(run_config["minimum_rows"])
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be a positive integer.")

    month = str(run_config["input_month"])
    if not re.fullmatch(r"20[0-9]{2}-[0-9]{2}", month):
        raise ValueError("input_month should look like 2024-01.")

    print("Validated taxi Python run configuration:")
    for key in sorted(run_config):
        print(f"{key}: {run_config[key]}")


def create_or_replace_config_map(core_api, namespace: str, name: str) -> None:
    config_map = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        data={"taxi_airflow_job.py": TAXI_SCRIPT_PATH.read_text()},
    )
    try:
        core_api.replace_namespaced_config_map(name, namespace, config_map)
        print(f"Replaced ConfigMap {namespace}/{name}")
    except ApiException as exc:
        if exc.status != 404:
            raise
        core_api.create_namespaced_config_map(namespace, config_map)
        print(f"Created ConfigMap {namespace}/{name}")


def submit_taxi_python_job(**context) -> None:
    run_config = merged_config(context)
    namespace = str(run_config["namespace"])
    name = run_name(context)
    secret_name = namespace
    config_map_name = f"{name}-code"

    config.load_incluster_config()
    core_api = client.CoreV1Api()
    batch_api = client.BatchV1Api()

    create_or_replace_config_map(core_api, namespace, config_map_name)

    container = client.V1Container(
        name="taxi-python",
        image=JOB_IMAGE,
        image_pull_policy="IfNotPresent",
        command=["bash", "-lc"],
        args=[
            "python -m pip install --no-cache-dir "
            + " ".join(JOB_DEPENDENCIES)
            + " && python "
            + f"{JOB_WORK_DIR}/taxi_airflow_job.py "
            + " ".join(
                [
                    "--shared-bucket",
                    str(run_config["shared_bucket"]),
                    "--output-bucket",
                    str(run_config["output_bucket"]),
                    "--input-month",
                    str(run_config["input_month"]),
                    "--minimum-rows",
                    str(run_config["minimum_rows"]),
                    "--output-prefix",
                    output_prefix(context),
                ]
            ),
        ],
        volume_mounts=[
            client.V1VolumeMount(
                name="chapter06-taxi-code",
                mount_path=JOB_WORK_DIR,
                read_only=True,
            )
        ],
        env=[
            client.V1EnvVar(
                name="AWS_ACCESS_KEY_ID",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name=secret_name,
                        key="AWS_ACCESS_KEY_ID",
                    )
                ),
            ),
            client.V1EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name=secret_name,
                        key="AWS_SECRET_ACCESS_KEY",
                    )
                ),
            ),
            client.V1EnvVar(name="S3_ENDPOINT_URL", value=str(run_config["s3_endpoint"])),
        ],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "3Gi"},
            limits={"memory": "6Gi"},
        ),
    )

    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        spec=client.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=3600,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[container],
                    volumes=[
                        client.V1Volume(
                            name="chapter06-taxi-code",
                            config_map=client.V1ConfigMapVolumeSource(
                                name=config_map_name
                            ),
                        )
                    ],
                )
            ),
        ),
    )

    try:
        batch_api.delete_namespaced_job(name, namespace, propagation_policy="Foreground")
        time.sleep(2)
    except ApiException as exc:
        if exc.status != 404:
            raise

    batch_api.create_namespaced_job(namespace=namespace, body=job)
    print(f"Submitted taxi Python Job {namespace}/{name}")
    print(f"Output prefix: s3://{run_config['output_bucket']}/{output_prefix(context)}")


def wait_for_taxi_python_job(**context) -> None:
    run_config = merged_config(context)
    namespace = str(run_config["namespace"])
    name = run_name(context)
    config.load_incluster_config()
    core_api = client.CoreV1Api()
    batch_api = client.BatchV1Api()
    deadline = time.time() + 30 * 60

    while time.time() < deadline:
        job = batch_api.read_namespaced_job(name=name, namespace=namespace)
        status = job.status
        if status.succeeded:
            print(f"Taxi Python Job {namespace}/{name} succeeded.")
            print_job_pod_logs(core_api, namespace, name)
            return
        if status.failed:
            print_job_pod_logs(core_api, namespace, name)
            raise RuntimeError(f"Taxi Python Job {namespace}/{name} failed.")
        print(f"Waiting for taxi Python Job {namespace}/{name}...")
        time.sleep(10)

    raise TimeoutError(f"Taxi Python Job {namespace}/{name} did not finish in time.")


def cleanup_taxi_config_map(**context) -> None:
    run_config = merged_config(context)
    namespace = str(run_config["namespace"])
    config_map_name = f"{run_name(context)}-code"
    config.load_incluster_config()
    core_api = client.CoreV1Api()
    try:
        core_api.delete_namespaced_config_map(config_map_name, namespace)
        print(f"Deleted ConfigMap {namespace}/{config_map_name}")
    except ApiException as exc:
        if exc.status != 404:
            raise
        print(f"ConfigMap {namespace}/{config_map_name} was already gone.")


def print_output_location(**context) -> None:
    run_config = merged_config(context)
    print("Taxi Python workflow finished.")
    print(f"Output: s3://{run_config['output_bucket']}/{output_prefix(context)}")
    print("Use the lab helper script to download and visualize this prefix.")


with DAG(
    dag_id="chapter06_taxi_python_pipeline",
    start_date=datetime.datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["preparation-for-ai", "chapter06", "python", "taxi"],
    doc_md=(
        "Chapter 6 lab DAG. Airflow creates a normal Kubernetes Python job "
        "that reads NYC taxi Parquet from object storage with Polars, builds "
        "small metric tables with DuckDB, and writes participant outputs to S3."
    ),
):
    start = EmptyOperator(task_id="start")
    validate_config = PythonOperator(
        task_id="validate_run_config",
        python_callable=validate_run_config,
    )
    submit_taxi_python_job_task = PythonOperator(
        task_id="submit_taxi_python_job",
        python_callable=submit_taxi_python_job,
    )
    wait_for_taxi_python_job_task = PythonOperator(
        task_id="wait_for_taxi_python_job",
        python_callable=wait_for_taxi_python_job,
    )
    cleanup_code = PythonOperator(
        task_id="cleanup_taxi_config_map",
        python_callable=cleanup_taxi_config_map,
    )
    show_output = PythonOperator(
        task_id="print_output_location",
        python_callable=print_output_location,
    )
    finish = EmptyOperator(task_id="finish")

    start >> validate_config >> submit_taxi_python_job_task
    submit_taxi_python_job_task >> wait_for_taxi_python_job_task
    wait_for_taxi_python_job_task >> cleanup_code >> show_output >> finish
