# Lab 6: Airflow, Orchestration, And Repeatability

In this lab you stop running every step by hand.

You use Airflow as the control layer for platform work. You trigger a workflow,
give it parameters, inspect the task graph, read logs, and check the evidence
written to object storage.

There are two workflows:

- Airflow submits a Python job for taxi analytics.
- Airflow submits a shared-GPU training job for video classification.

Airflow coordinates the work. Polars, DuckDB, PyTorch, Kubernetes, and object
storage still do their own jobs.

That separation is the point. Airflow should not become the place where heavy
data processing or model training happens. It should describe the workflow,
submit work to the right compute system, wait for the result, and make success
or failure visible.

The challenge in this chapter is to follow evidence across systems. A run
starts in Airflow, creates external compute, writes outputs to object storage,
and leaves logs behind. Your job is to connect those pieces.

## Log In To Airflow

Open Airflow:

```text
http://airflow.mechatronics.lan
```

Your VM already has your Airflow account in `participant.env`:

```bash
grep '^AIRFLOW_' participant.env
```

Use `AIRFLOW_USERNAME` and `AIRFLOW_PASSWORD` to log in. Airflow is shared, but
each participant uses a separate account so runs and audit trails are easier to
follow during the workshop.

In Airflow, a DAG is a workflow. A task is one step inside that workflow. The
graph view shows the order of the tasks and the state of each task in a run.

The Airflow interface has a few places you will use repeatedly:

- `Dags` in the left sidebar lists workflows.
- the search box at the top of the `Dags` page filters the list.
- the `Trigger` button is the small play-triangle button in the top right of
  the pipeline box. It starts a manual run.
- `Advanced Options` in the trigger dialog contains the `Configuration JSON`
  box where you paste run parameters.
- after opening a DAG, the `Runs` tab shows past runs of that DAG.
- a run page shows one row per task. Click a task row and open `Logs` to see
  what happened.

The relationship looks like this:

```text
DAG / workflow
  -> DAG run / one execution with one configuration
    -> task instance / one task inside that run
      -> logs / what that task did and why it succeeded or failed
      -> output evidence / paths written to object storage
```

## Lab 1: Airflow Submits A Python Analytics Job

In Airflow, click `Dags` in the left sidebar. Use the search box to find this
DAG, then click its name:

```text
chapter06_taxi_python_pipeline
```

Also open the readable course copy:

```text
06-orchestration/dags/chapter06_taxi_python_pipeline.py
```

This workflow reads one month of NYC Yellow Taxi Parquet data from the shared
bucket. Airflow creates a normal Kubernetes Job. That job runs a Python script:
Polars cleans the taxi trips, DuckDB SQL builds Gold metric tables, and the job
writes the outputs to your participant bucket.

In other words, Airflow is not doing the taxi analysis itself. It is acting like
the control room: validate the run, start the job, watch the job, and point you
to the evidence.

Before you trigger it, inspect the DAG file and find these tasks:

- `validate_run_config`
- `submit_taxi_python_job`
- `wait_for_taxi_python_job`
- `cleanup_taxi_config_map`
- `print_output_location`

The important pattern is:

```text
Airflow validates parameters
  -> Airflow creates a Kubernetes Job
  -> Python reads shared object storage
  -> Polars and DuckDB write participant outputs
  -> Airflow waits and records the result
```

Click the small play-triangle `Trigger` button in the top right of the pipeline
box. Keep `Single Run` selected.
Open `Advanced Options`. Leave `Logical Date`, `Run ID`, and `Partition key`
unchanged, but replace the contents of `Configuration JSON` with this:

```json
{
  "participant": "preparation-for-ai-XX",
  "namespace": "preparation-for-ai-XX",
  "shared_bucket": "preparation-for-ai-shared",
  "output_bucket": "preparation-for-ai-XX",
  "input_month": "2024-01",
  "minimum_rows": 100000
}
```

Replace `preparation-for-ai-XX` with your participant name.

Each configuration value has a job:

- `participant` identifies you in reports and logs.
- `namespace` tells Airflow where your Kubernetes resources are created.
- `shared_bucket` contains the input dataset.
- `output_bucket` is where your results are written.
- `input_month` chooses the taxi month to process.
- `minimum_rows` is a quality check. If the cleaned dataset is smaller than
  this number, the workflow should fail.

Click the blue `Trigger` button at the bottom of the dialog.

Follow the run in Airflow:

1. Open the `Runs` tab.
2. Click the newest `Dag Run ID`.
3. Look at the task table. A successful run should show green `Success` states.
4. Click `validate_run_config`, then open `Logs`. Check that Airflow printed
   the configuration you pasted.
5. Click `submit_taxi_python_job`, then open `Logs`. Check that Airflow created
   a Kubernetes job.
6. Click `wait_for_taxi_python_job`, then open `Logs`. Check that Airflow waited
   for the Kubernetes job to finish. After the job finishes, this same log also
   includes the Kubernetes pod logs from the Python job.
7. Click `print_output_location`, then open `Logs`. Copy the `Output:
   s3://...` line.

Do not skip the logs. In this lab the logs are not only debugging noise; they
are the operational evidence that Airflow submitted external compute and waited
for it to finish.

The most useful log is often `wait_for_taxi_python_job`. It shows both layers:
Airflow waiting for Kubernetes, and then the full stdout/stderr from the
Kubernetes pod that actually ran the taxi code.

The output prefix looks like this:

```text
outputs/chapter06/taxi-python/run_id=...
```

Use the part after your bucket name. For example, if Airflow prints:

```text
s3://preparation-for-ai-01/outputs/chapter06/taxi-python/run_id=manual-example/
```

then the prefix you need is:

```text
outputs/chapter06/taxi-python/run_id=manual-example/
```

Copy that prefix into:

```text
06-orchestration/lab1/taxi-python-outputs.py
```

Fill in:

```python
RUN_PREFIX = "outputs/chapter06/taxi-python/run_id=..."
```

Run:

```bash
source .venv/bin/activate
python3 06-orchestration/lab1/taxi-python-outputs.py
```

The helper script writes four local files:

```text
06-orchestration/lab1/taxi1_output_summary.md
06-orchestration/lab1/taxi2_hourly_metrics.png
06-orchestration/lab1/taxi3_payment_metrics.png
06-orchestration/lab1/taxi4_distance_metrics.png
```

Open them in order.

First open:

```text
06-orchestration/lab1/taxi1_output_summary.md
```

This summary is the bridge between Airflow and object storage. It shows the
run prefix you copied, the report text, and the objects written by the job. Find
the Silver Parquet file and the Gold metric tables. Those paths are the
evidence that the DAG did useful data-platform work.

Then open:

```text
06-orchestration/lab1/taxi2_hourly_metrics.png
```

This plot shows trips and average tip rate by pickup hour. The point is not
only the taxi pattern. The point is that Airflow triggered an external job, the
job created a Gold metric table, and your local helper script turned it into
something inspectable.

Then open:

```text
06-orchestration/lab1/taxi3_payment_metrics.png
```

This plot compares payment types using several measurements. Credit card should
dominate the trip count. The average values are useful because a workflow can
produce more than one kind of evidence: row counts, business metrics, and
validation checks.

Notice the tip-rate panel. It may look odd that cash trips have no visible tip
rate. That is a data interpretation detail: in this taxi dataset, the recorded
tip amount mostly represents electronic tips, so cash tips are not captured in
the same way. A dashboard or model would need that context.

Finally open:

```text
06-orchestration/lab1/taxi4_distance_metrics.png
```

This plot groups trips into distance buckets. Check whether most trips are short
and whether longer trips have higher average total amounts. This is the kind of
small question a recurring analytics workflow can answer every day.

Use the outputs to answer concrete questions:

- Which hours have the most taxi trips?
- Does the average tip rate change by hour?
- Which payment types dominate this month?
- Do longer distance buckets cost more on average?
- Which object-storage object contains the Silver data?
- Which object-storage objects contain the Gold metric tables?

Those questions force you to connect the Airflow run to the data lake outputs.
If you can answer them, you understand more than "the DAG turned green."

Now trigger a failing run from the same DAG page. Click `Trigger`, open
`Advanced Options`, paste the same configuration, but set:

```json
"minimum_rows": 999999999
```

The Python job should fail because the cleaned dataset does not have that many
rows. Open the `Runs` tab again, click the failed run, and find the red task.
Open that task's `Logs` tab and find the exact error message.

Then rerun with a realistic `minimum_rows`.

This is the point of orchestration: a data-quality failure should stop the
workflow in a visible place, not disappear inside someone else's terminal.

## Lab 2: Airflow Submits Shared-GPU Training

Go back to `Dags` in the left sidebar. Use the search box to find this DAG,
then click its name:

```text
chapter06_video_gpu_pipeline
```

Also open the readable course copy:

```text
06-orchestration/dags/chapter06_video_gpu_pipeline.py
```

This workflow trains a small video action-recognition model on shared GPU
compute. The dataset is stored as video clips in object storage plus a manifest
table. Airflow creates a Kubernetes GPU job. The training job reads the
manifest, loads the video clips, trains the PyTorch model, and writes model
evidence back to your participant bucket.

The workshop platform currently exposes one shared full GPU to Kubernetes. That
means one GPU training pod can run at a time. If several participants trigger
this DAG together, Kubernetes queues the extra GPU jobs until the GPU is free.
That is expected. In many teams, a powerful GPU is a shared resource. People do
not SSH into it manually; they submit jobs, the platform schedules them, and the
workflow records what happened.

So if your GPU run waits for a while, do not immediately assume it is broken.
Open the task logs and look for the waiting messages. That queue is part of the
lesson: Airflow plus Kubernetes gives the team a controlled way to share scarce
compute. A normal run takes a few minutes even when it starts immediately,
because the job installs Python packages, downloads video clips, trains for five
epochs, and uploads the evidence files.

Before you trigger it, inspect the DAG file and find these tasks:

- `validate_run_config`
- `submit_gpu_training_job`
- `wait_for_gpu_training_job`

The important pattern is:

```text
Airflow validates parameters
  -> Airflow creates a Kubernetes GPU job
  -> the GPU job trains a PyTorch model
  -> the job writes evidence to object storage
  -> Airflow waits and records the result
```

Click the small play-triangle `Trigger` button in the top right of the pipeline
box. Keep `Single Run` selected.
Open `Advanced Options`. Leave the date and run fields unchanged, but replace
`Configuration JSON` with this:

```json
{
  "participant": "preparation-for-ai-XX",
  "namespace": "preparation-for-ai-XX",
  "shared_bucket": "preparation-for-ai-shared",
  "output_bucket": "preparation-for-ai-XX",
  "epochs": 5,
  "batch_size": 4,
  "max_validation_error": 0.90
}
```

The configuration controls both the run location and the training behavior:

- `participant` identifies you in reports and logs.
- `namespace` tells Airflow where your Kubernetes job is created.
- `shared_bucket` contains the video clips and manifest.
- `output_bucket` is where your results are written.
- `epochs` controls how many passes the model makes over the training data.
- `batch_size` controls how many clips are processed together.
- `max_validation_error` is a validation gate. If the model error is too high,
  the workflow should fail instead of pretending the run is good.

The model is TorchVision's pretrained `r3d_18`, a 3D convolutional video model.
It sees 16 frames from each clip, not only one still image. The prediction
output shows a few frames so you can inspect the clips visually, but the model
itself receives a short frame sequence.

The `max_validation_error` value of `0.90` is deliberately forgiving. It means
"allow the run to pass unless the model is extremely bad." The dataset is still
small and the model only trains briefly, so the goal of the first run is to
prove that Airflow, Kubernetes, CUDA, object storage, and PyTorch all work
together.

Click the blue `Trigger` button at the bottom of the dialog.

Follow the run in Airflow:

1. Open the `Runs` tab.
2. Click the newest `Dag Run ID`.
3. Click `validate_run_config`, then open `Logs`. Check the epoch count, batch
   size, manifest key, and validation gate.
4. Click `submit_gpu_training_job`, then open `Logs`. This is where Airflow
   prints the Kubernetes job name and the output prefix.
5. Click `wait_for_gpu_training_job`, then open `Logs`. This task should print
   repeated waiting messages until the GPU job succeeds. After the job
   finishes, the same Airflow log includes the Kubernetes pod logs from the GPU
   training container.

The output prefix looks like this:

```text
outputs/chapter06/video-gpu/run_id=...
```

Find it in the `submit_gpu_training_job` task logs. Look for a line like:

```text
Output prefix: s3://preparation-for-ai-01/outputs/chapter06/video-gpu/run_id=manual-example/
```

Copy only the part after your bucket name.

Copy that prefix into:

```text
06-orchestration/lab2/video-gpu-outputs.py
```

Fill in:

```python
RUN_PREFIX = "outputs/chapter06/video-gpu/run_id=..."
```

Run:

```bash
source .venv/bin/activate
python3 06-orchestration/lab2/video-gpu-outputs.py
```

The helper script downloads the GPU job evidence into the lab folder:

```text
06-orchestration/lab2/video1_dataset_overview.png
06-orchestration/lab2/video2_training_curve.png
06-orchestration/lab2/video3_confusion_matrix.png
06-orchestration/lab2/video4_predictions.png
06-orchestration/lab2/video5_preview_*.avi
```

It also prints the full object list in the terminal. That terminal output is
useful evidence that the files came from object storage.

Open them in order.

First open:

```text
06-orchestration/lab2/video1_dataset_overview.png
```

This plot shows the video-action classes in the shared dataset. It should now
contain more than a tiny four-class example. Check the train/validation counts
and notice that clip size varies by action class.

Then open:

```text
06-orchestration/lab2/video2_training_curve.png
```

The curve shows training loss, validation accuracy, and validation error across
five epochs. This is the fastest visual check of the run.

Then open:

```text
06-orchestration/lab2/video3_confusion_matrix.png
```

Rows are true action labels. Columns are predicted action labels. Off-diagonal
counts show which actions the model confuses.

Finally open:

```text
06-orchestration/lab2/video4_predictions.png
```

This image shows several validation clips per class. Each example contains the
first, middle, and last frame of the clip, with the predicted label below it.
That is still only a preview of the video, but it should feel less like an image
classifier and more like a video dataset.

Then open a few files named:

```text
06-orchestration/lab2/video5_preview_*.avi
```

These are actual moving video clips copied from the validation data. The still
prediction sheet is easier to compare quickly, but the video files make the
modality real: the model is learning from short actions over time, not only from
one photograph.

Use the outputs to answer concrete questions:

- Did training loss decrease?
- Did validation accuracy improve across the five epochs?
- Which action classes are confused most often?
- Do the example video predictions look believable?
- Which task proves that the GPU job finished?
- Which file is the clearest evidence that the model should be trusted or not?

Now trigger a controlled failure from the same DAG page. Click `Trigger`, open
`Advanced Options`, paste the same configuration, but make the validation gate
stricter:

```json
"max_validation_error": 0.01
```

The model is unlikely to reach that error threshold in a short workshop run.
Open the failed run from the `Runs` tab, click the failed task, and read the
error message in `Logs`.

Then rerun with a realistic `max_validation_error`.

This controlled failure is important. In a real platform, orchestration is not
only about successful runs. It is also about making bad runs visible early
enough that the team does not trust the wrong output.

## What You Should Notice

Airflow is not the data lake, the query engine, or the model framework.

Airflow answers operational questions:

- What should run?
- In what order?
- With which parameters?
- Did it succeed?
- Where are the logs?
- Where is the evidence?

The compute can change. In this chapter, one workflow uses a normal Python data
job and one uses a GPU training job. The orchestration pattern stays the same:

```text
trigger
  -> validate
  -> submit external compute
  -> wait
  -> inspect evidence
```

That pattern is the main lesson of this chapter.

The question to carry into the next chapters is:

```text
Where would I look if this workflow failed tomorrow morning?
```

If the answer is visible in Airflow logs, Kubernetes state, and object-storage
evidence, the workflow is becoming operational instead of only experimental.
