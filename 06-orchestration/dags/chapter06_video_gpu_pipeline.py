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

DEFAULT_CONFIG = {
    "namespace": "preparation-for-ai-dev",
    "shared_bucket": "preparation-for-ai-shared",
    "output_bucket": "preparation-for-ai-dev",
    "manifest_key": "datasets/chapter06/video-actions/manifest.csv",
    "epochs": 5,
    "batch_size": 4,
    "max_validation_error": 0.90,
}

VIDEO_TRAINING_SCRIPT_PATH = Path(
    os.getenv(
        "CHAPTER06_VIDEO_TRAINING_SCRIPT_PATH",
        "/opt/airflow/jobs/video_action_training.py",
    )
)

TRAINING_IMAGE = "pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime"
GPU_RESOURCE = "nvidia.com/gpu"
TRAINING_WORK_DIR = "/opt/course"
TRAINING_DEPENDENCIES = [
    "boto3==1.43.6",
    "pandas==2.3.3",
    "scikit-learn==1.8.0",
    "matplotlib==3.10.8",
    "pillow==12.2.0",
    "av==17.0.1",
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
    return dns_name(f"chapter06-video-{context['run_id']}", 52)


def output_prefix(context) -> str:
    return f"outputs/chapter06/video-gpu/run_id={dns_name(context['run_id'], 80)}/"


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
    for name in ["namespace", "shared_bucket", "output_bucket", "manifest_key"]:
        if not run_config.get(name):
            raise ValueError(f"Missing run configuration value: {name}")

    epochs = int(run_config["epochs"])
    batch_size = int(run_config["batch_size"])
    max_validation_error = float(run_config["max_validation_error"])
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not 0 <= max_validation_error <= 1:
        raise ValueError("max_validation_error should be between 0 and 1.")

    print("Validated video GPU run configuration:")
    for key in sorted(run_config):
        print(f"{key}: {run_config[key]}")


def submit_gpu_training_job(**context) -> None:
    run_config = merged_config(context)
    namespace = str(run_config["namespace"])
    name = run_name(context)
    secret_name = namespace
    config_map_name = f"{name}-code"

    config.load_incluster_config()
    core_api = client.CoreV1Api()
    batch_api = client.BatchV1Api()

    config_map = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=config_map_name, namespace=namespace),
        data={"video_action_training.py": VIDEO_TRAINING_SCRIPT_PATH.read_text()},
    )
    try:
        core_api.replace_namespaced_config_map(config_map_name, namespace, config_map)
        print(f"Replaced ConfigMap {namespace}/{config_map_name}")
    except ApiException as exc:
        if exc.status != 404:
            raise
        core_api.create_namespaced_config_map(namespace, config_map)
        print(f"Created ConfigMap {namespace}/{config_map_name}")

    container = client.V1Container(
        name="video-training",
        image=TRAINING_IMAGE,
        image_pull_policy="IfNotPresent",
        command=["bash", "-lc"],
        args=[
            "python -m pip install --break-system-packages --no-cache-dir "
            + " ".join(TRAINING_DEPENDENCIES)
            + " && python "
            + f"{TRAINING_WORK_DIR}/video_action_training.py "
            + " ".join(
                [
                    "--shared-bucket",
                    str(run_config["shared_bucket"]),
                    "--output-bucket",
                    str(run_config["output_bucket"]),
                    "--manifest-key",
                    str(run_config["manifest_key"]),
                    "--output-prefix",
                    output_prefix(context),
                    "--epochs",
                    str(run_config["epochs"]),
                    "--batch-size",
                    str(run_config["batch_size"]),
                    "--max-validation-error",
                    str(run_config["max_validation_error"]),
                ]
            ),
        ],
        volume_mounts=[
            client.V1VolumeMount(
                name="chapter06-video-code",
                mount_path=TRAINING_WORK_DIR,
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
            client.V1EnvVar(name="S3_ENDPOINT_URL", value="http://s3.mechatronics.lan"),
        ],
        resources=client.V1ResourceRequirements(
            limits={GPU_RESOURCE: "1"},
            requests={"cpu": "2", "memory": "8Gi", GPU_RESOURCE: "1"},
        ),
    )

    pod_spec = client.V1PodSpec(
        restart_policy="Never",
        containers=[container],
        tolerations=[
            client.V1Toleration(
                key="nvidia.com/gpu",
                operator="Equal",
                value="true",
                effect="NoSchedule",
            )
        ],
        volumes=[
            client.V1Volume(
                name="chapter06-video-code",
                config_map=client.V1ConfigMapVolumeSource(name=config_map_name),
            )
        ],
    )
    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        spec=client.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=3600,
            template=client.V1PodTemplateSpec(spec=pod_spec),
        ),
    )

    try:
        batch_api.delete_namespaced_job(
            name,
            namespace,
            propagation_policy="Foreground",
        )
        time.sleep(2)
    except ApiException as exc:
        if exc.status != 404:
            raise

    batch_api.create_namespaced_job(namespace=namespace, body=job)
    print(f"Submitted GPU training Job {namespace}/{name}")
    print(f"Output prefix: s3://{run_config['output_bucket']}/{output_prefix(context)}")


def wait_for_gpu_training_job(**context) -> None:
    run_config = merged_config(context)
    namespace = str(run_config["namespace"])
    name = run_name(context)
    config.load_incluster_config()
    core_api = client.CoreV1Api()
    batch_api = client.BatchV1Api()
    deadline = time.time() + 60 * 60

    while time.time() < deadline:
        job = batch_api.read_namespaced_job(name=name, namespace=namespace)
        status = job.status
        if status.succeeded:
            print(f"GPU training Job {namespace}/{name} succeeded.")
            print_job_pod_logs(core_api, namespace, name)
            return
        if status.failed:
            print_job_pod_logs(core_api, namespace, name)
            raise RuntimeError(f"GPU training Job {namespace}/{name} failed.")
        print(f"Waiting for GPU training Job {namespace}/{name}...")
        time.sleep(15)

    raise TimeoutError(f"GPU training Job {namespace}/{name} did not finish in time.")


def cleanup_training_config_map(**context) -> None:
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


with DAG(
    dag_id="chapter06_video_gpu_pipeline",
    start_date=datetime.datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["preparation-for-ai", "chapter06", "gpu", "video"],
    doc_md=(
        "Chapter 6 lab DAG for shared-GPU video model training. Airflow "
        "creates a Kubernetes GPU job, waits for completion, and keeps the "
        "run visible in one workflow."
    ),
):
    start = EmptyOperator(task_id="start")
    validate_config = PythonOperator(
        task_id="validate_run_config",
        python_callable=validate_run_config,
    )
    submit_training = PythonOperator(
        task_id="submit_gpu_training_job",
        python_callable=submit_gpu_training_job,
    )
    wait_for_training = PythonOperator(
        task_id="wait_for_gpu_training_job",
        python_callable=wait_for_gpu_training_job,
    )
    cleanup_config_map = PythonOperator(
        task_id="cleanup_training_config_map",
        python_callable=cleanup_training_config_map,
    )
    finish = EmptyOperator(task_id="finish")

    start >> validate_config >> submit_training >> wait_for_training
    wait_for_training >> cleanup_config_map >> finish
