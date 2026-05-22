# Chapter 6: Airflow, Orchestration, And Repeatability

Up to this point, you have mostly started the work yourself.

You ran Python scripts. You submitted compute jobs. You inspected object storage
outputs. That is the right way to learn the pieces, because you need to
understand what each tool is doing.

It is not enough for a shared data platform.

A production workflow cannot depend on one developer remembering which command
to run, from which directory, with which environment variables. If the work has
to happen every morning, after new files arrive, or before a model is retrained,
the control layer should be part of the platform.

That is what orchestration adds.

An orchestrator runs workflow steps in the right order, records what happened,
and makes failures visible to the team.

It usually provides:

- dependency graphs
- manual triggers
- schedules
- retries
- logs
- parameters
- run history
- a user interface for operational visibility

In this chapter, you use Apache Airflow as that shared control layer.

## From Manual Runs To Platform Runs

A manual data preparation workflow often looks like this:

```text
open terminal
  -> run Python script
  -> inspect output
  -> run SQL or Python job
  -> inspect output again
  -> train or evaluate a model
  -> copy metrics somewhere useful
```

That workflow is good for exploration. It is weak as an operational pattern.
The knowledge lives in the developer's head and terminal history.

An orchestrated workflow moves the control layer into a service:

```text
trigger a run
  -> task A prepares or checks input data
  -> task B builds a reusable dataset
  -> task C runs heavier compute when needed
  -> task D writes evidence and metrics
  -> logs and run state remain visible afterwards
```

The important change is not that Airflow magically makes the code better. The
important change is that the workflow becomes explicit, repeatable, and visible
to other people.

## What Airflow Is

Airflow is a workflow orchestrator.

It is not a dataframe library. It is not a database. It is not a model
framework. It is not the place where you should hide all transformation logic.

Airflow coordinates work.

The core object in Airflow is a DAG: a directed acyclic graph. In practical
terms, a DAG is a workflow definition.

```text
task_a -> task_b -> task_c
```

The arrows define dependencies. `task_b` should not start until `task_a`
finishes. `task_c` should not start until `task_b` finishes.

Airflow keeps track of:

- which DAGs exist
- when they should run
- which task instances succeeded or failed
- where task logs are stored
- whether failed tasks should be retried
- which parameters were used for a run

This is why Airflow is useful even when every task is still "just Python" or
"just SQL". The value is in the operational wrapper around the work.

## DAGs, Tasks, And Operators

An Airflow DAG is written as Python code.

A small DAG can look like this:

```python
import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG

with DAG(
    dag_id="chapter06_example",
    start_date=datetime.datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
):
    start = EmptyOperator(task_id="start")

    inspect_environment = BashOperator(
        task_id="inspect_environment",
        bash_command="date -Iseconds && python --version",
    )

    start >> inspect_environment
```

There are a few ideas in this example.

`dag_id` is the workflow name. It should be stable and readable.

`schedule=None` means the workflow does not run automatically. You trigger it
manually from the user interface or API. That is useful in a workshop.

`catchup=False` prevents Airflow from trying to run old scheduled intervals
when a DAG is enabled. That matters for scheduled production DAGs.

An operator defines what a task does. `BashOperator` runs a shell command.
`EmptyOperator` is useful as a marker or dependency point. Other operators can
run Python functions, submit Kubernetes jobs, call APIs, or trigger external
systems.

The DAG file should stay lightweight. The heavy data logic should live in
normal scripts, SQL files, or container images that can be tested outside
Airflow.

## Airflow In This Workshop

The workshop uses a shared Airflow service.

You do not install Airflow yourself. You use it as a platform user:

- open the Airflow user interface
- inspect a DAG
- trigger a run
- provide run parameters
- follow task state
- inspect logs
- find the output in object storage

That is the professional view most data scientists need first.

Airflow itself runs as shared infrastructure. Your participant bucket remains
your output location. When heavier processing is needed, Airflow can coordinate
external compute such as Kubernetes jobs.

The pattern is:

```text
Airflow controls the workflow
object storage holds inputs and outputs
Python, SQL, or PyTorch does the compute
reports make the run inspectable
```

Airflow is the control plane, not the data lake.

## Run Parameters

Parameters make one DAG reusable.

Without parameters, every participant, dataset, date, or model variant would
need a separate workflow definition. That does not scale.

A run can receive values such as:

- participant id
- source dataset
- output bucket
- output prefix
- processing date
- row-count threshold
- number of training epochs
- validation metric threshold

In this workshop, parameters are useful because every participant has a
separate object-storage bucket and Kubernetes namespace. The same DAG structure
can run for different participants without mixing outputs.

A manual trigger configuration can look like this:

```json
{
  "participant": "preparation-for-ai-01",
  "namespace": "preparation-for-ai-01",
  "shared_bucket": "preparation-for-ai-shared",
  "output_bucket": "preparation-for-ai-01",
  "input_month": "2024-01",
  "minimum_rows": 100000
}
```

The DAG code can read those values and use them to choose input and output
locations. That is how one shared workflow can still produce isolated
participant outputs.

Inside the DAG, `dag_run.conf` contains the configuration that was passed when
the run was triggered. A small helper function can merge those values with
defaults:

```python
DEFAULT_CONFIG = {
    "participant": "preparation-for-ai-XX",
    "namespace": "preparation-for-ai-XX",
    "shared_bucket": "preparation-for-ai-shared",
    "output_bucket": "preparation-for-ai-XX",
    "input_month": "2024-01",
    "minimum_rows": 100000,
}


def merged_config(context):
    run_config = dict(DEFAULT_CONFIG)
    run_config.update(context["dag_run"].conf or {})
    return run_config
```

That small piece of code connects the manual trigger form in the user interface
to the tasks that do the work.

## Orchestration Is Not Transformation

It is tempting to put all logic directly inside an Airflow DAG file.

That becomes difficult to maintain.

A better pattern is:

```text
Airflow DAG
  -> calls clear task code
  -> task code reads from object storage
  -> task code writes outputs and evidence
  -> Airflow records run state and logs
```

The DAG should answer operational questions:

- What steps exist?
- In what order should they run?
- What parameters does the run need?
- What should happen when a task fails?
- Where can the team inspect logs?

The task code should answer data questions:

- Which files or tables are read?
- Which transformations happen?
- Which outputs are written?
- Which metrics or reports prove the step worked?

Keeping those responsibilities separate makes the workflow easier to test.
You can run a transformation script directly while developing it. Later,
Airflow can call the same script as one task in a larger workflow.

For example, the transformation code can stay in a normal script, while the DAG
only contains the code that starts and monitors the work:

```python
submit_job = PythonOperator(
    task_id="submit_taxi_python_job",
    python_callable=submit_taxi_python_job,
)
```

The DAG says when the job should run. The job script says how the taxi data is
cleaned and summarized.

## Python Jobs From Airflow

The first lab uses a normal Python job.

That may sound less dramatic than a distributed compute system, but it is the
right first orchestration pattern. Many useful workflows are ordinary Python
programs that need to run reliably with the right inputs, credentials,
parameters, and logs.

The mental model is:

```text
Airflow task
  -> creates a Kubernetes Job
  -> waits for success or failure
  -> the job reads from object storage
  -> Polars and DuckDB do the data work
  -> the job writes results back to object storage
  -> Airflow records the task state
```

This is where orchestration and shared compute meet.

Airflow does not need to become the dataframe engine. It only needs to start the
work, wait for it, and record whether the workflow can continue.

In a larger setup, a DAG might:

- check that a manifest exists
- submit a Python or SQL job to build a Gold table
- wait until the job succeeds
- run a small metric query
- publish a report
- stop before training if the data is not acceptable

The same idea applies to other systems. Airflow can call a Python script, a SQL
job, a container, a cloud service, or a model-training endpoint. The important
part is that each task has clear inputs, outputs, and failure behavior.

In the workshop DAG, Airflow uses the Kubernetes Python client to create a
`Job` object. The task code is longer than this simplified example, but the
shape is the same:

```python
batch_api.create_namespaced_job(
    namespace=namespace,
    body=job,
)
```

Another task polls the `Job` until it reaches a terminal state:

```python
if job.status.succeeded:
    return
if job.status.failed:
    raise RuntimeError("Taxi Python job failed.")
```

The important idea is concrete: Airflow starts external compute, waits for a
clear success or failure signal, and records the logs in one visible workflow.

## GPU Jobs From Airflow

Data preparation is not the only external compute Airflow can coordinate.

A machine-learning workflow may need GPU compute for model training. In this
course, the GPU training lab uses the same orchestration pattern, but the
external compute is a Kubernetes Job that requests a GPU resource.

The mental model is:

```text
Airflow task
  -> creates a Kubernetes Job
  -> Kubernetes schedules a GPU pod
  -> PyTorch trains the model
  -> the job writes plots and preview clips to object storage
  -> Airflow records success or failure
```

The GPU job receives normal run parameters: input manifest, output bucket,
number of epochs, batch size, and a validation threshold. The training code is
still PyTorch. Airflow only decides when the training should start, which
parameters it should use, and whether the result is acceptable.

A simplified container definition looks like this:

```python
container = client.V1Container(
    name="video-training",
    image="pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime",
    args=[
        "--shared-bucket",
        run_config["shared_bucket"],
        "--output-bucket",
        run_config["output_bucket"],
        "--epochs",
        str(run_config["epochs"]),
    ],
    resources=client.V1ResourceRequirements(
        requests={"cpu": "2", "memory": "8Gi", "nvidia.com/gpu": "1"},
        limits={"nvidia.com/gpu": "1"},
    ),
)
```

The resource request is the important Kubernetes detail. It tells Kubernetes
that this pod needs GPU capacity. In a larger platform, that GPU capacity might
come from a whole GPU or from a partitioned GPU resource. The platform handles
that scheduling detail; the workflow records the run.

In this workshop, the Kubernetes cluster exposes one full shared GPU. A job that
requests `nvidia.com/gpu: 1` uses that whole GPU while it runs. If multiple
participants trigger the workflow at the same time, Kubernetes queues the extra
GPU jobs until the resource is free. That queueing is normal shared-platform
behavior, and Airflow still gives each run a visible state, logs, and output
location.

## Quality Checks As Workflow Steps

A check that only runs when a developer remembers it is weak.

A check that runs inside the workflow is stronger.

For example, before a training task starts, a workflow can check:

- the expected manifest exists
- referenced assets exist
- class counts are not empty
- train and test splits do not overlap by group id
- a Parquet output contains the expected columns
- a metric is within an acceptable range

These checks can be normal Python code. They can also be organized with
specialized tools.

Pandera is useful when you want dataframe schema checks in Python.

Great Expectations is useful when a team wants more explicit data-quality
expectations and reports.

dbt tests are useful when a team builds SQL transformation models and wants
checks close to those models.

The tool is less important than the placement. The check should run before bad
data moves forward to a dashboard, a model, or a business decision.

A check can be a normal task in the DAG:

```python
validate_config = PythonOperator(
    task_id="validate_run_config",
    python_callable=validate_run_config,
)

validate_config >> submit_job
```

If the check exits with an error, Airflow marks the task as failed and the
downstream task does not run.

## Idempotent Tasks

An orchestrated task should be safe to retry.

Airflow can rerun a failed task. That is useful when the failure was temporary:
network hiccup, object storage timeout, busy cluster, or a worker restart.

Retries are dangerous when tasks are not designed carefully.

A task is idempotent when running it twice with the same inputs produces the
same intended result.

For data preparation, that usually means:

- write outputs to a predictable location for the run
- overwrite only when that is an explicit choice
- avoid appending duplicate rows on retry
- include the run id or dataset version in reports
- make partial outputs detectable
- fail clearly when required inputs are missing

Object storage makes this both easier and harder. It is easy to write files to
a new prefix. It is also easy to leave behind partial outputs if a task fails
halfway through.

That is why many workflows write evidence files:

```text
outputs/chapter06/taxi-python/run_id=manual-example/taxi1_run_report.md
outputs/chapter06/taxi-python/run_id=manual-example/gold/hourly_metrics.parquet
outputs/chapter06/video-gpu/run_id=manual-example/video2_training_curve.png
```

Those files help the team understand which input was used, which output was
created, whether the run should be trusted, and where to look next.

The DAG can pass the run id into the output prefix:

```python
def output_prefix(context):
    return f"outputs/chapter06/taxi-python/run_id={dns_name(context['run_id'])}/"
```

That does not solve every retry problem, but it avoids one common mistake:
different runs silently writing over each other.

## Schedules And Backfills

Airflow DAGs can run manually or on a schedule.

Manual runs are useful for workshops, experiments, and one-off operational
tasks.

Scheduled runs are useful when the data arrives predictably:

```text
every morning at 06:00
every Monday
after the previous daily partition should be available
```

In code, a daily schedule can be expressed directly on the DAG:

```python
with DAG(
    dag_id="daily_manifest_pipeline",
    start_date=datetime.datetime(2026, 1, 1),
    schedule="0 6 * * *",
    catchup=False,
):
    ...
```

`schedule="0 6 * * *"` means 06:00 every day. The syntax is cron syntax, which
is common in scheduling systems.

Backfills are historical runs. If a daily pipeline failed for three days, you
may need to rerun those three data intervals. If a transformation bug is fixed,
you may need to rebuild an older range of outputs.

Backfills are powerful because the same DAG definition can process old data
again. They are also risky when tasks are not idempotent. If a task appends
duplicate rows or overwrites the wrong prefix, rerunning history can make the
dataset worse instead of better.

That is why idempotency and scheduling belong together in production thinking.
If a workflow will run repeatedly, every task needs clear rules for where it
reads, where it writes, and what happens when it runs again.

## Airflow, Dagster, And Prefect

Airflow is not the only orchestrator.

Dagster and Prefect are also common in modern data teams.

Dagster has a strong asset-oriented mental model. It is designed around the
idea that workflows produce data assets, and that those assets should be
observable.

Prefect is Python-first and often feels approachable for teams that want a
lighter workflow experience.

Airflow is the hands-on tool in this course because it is mature, widely
recognized, and already deployed as shared infrastructure for the workshop.

The concepts transfer:

```text
workflow definition
tasks
dependencies
parameters
logs
retries
run history
```

Different tools use different names and APIs, but the operational problem is
the same.

## Lab 6 Preview

In the labs, you use Airflow as a platform user.

The first workflow submits a Python analytics job. Airflow validates the run
configuration, creates a Kubernetes Job, waits for it to finish, and prints the
output location. The job reads NYC taxi Parquet from shared object storage,
uses Polars and DuckDB to create Silver and Gold outputs, and writes them to
the participant bucket.

The second workflow submits shared-GPU training. Airflow validates the run
configuration, creates a Kubernetes training job, waits for it to finish, and
keeps the logs visible. The training job reads video clips from object storage,
fine-tunes a PyTorch action-recognition model, and writes model evidence back
to the participant bucket: a dataset overview, training curves, a confusion
matrix, prediction examples, and a few moving video previews.

The lab focuses on:

- reading the DAG structure
- triggering a run
- passing participant parameters
- inspecting task logs
- understanding external compute
- understanding retry behavior
- finding evidence in object storage
- making a run fail in a controlled way
- rerunning with corrected parameters

The goal is not to become an Airflow administrator. The goal is to understand
how repeatable data preparation becomes a shared platform workflow.
