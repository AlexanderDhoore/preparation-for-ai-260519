# Chapter 8: Practical MLOps And Reproducibility

The previous chapters prepared data for machine learning.

This final chapter asks what happens after a model is trained.

A model result is not very useful if it only exists as a number printed in a
terminal. A professional team needs evidence:

- which dataset trained the model
- which parameters were used
- which metrics were produced
- which plots or reports explain the result
- where the trained model artifact is stored
- how another system could use the model

That evidence is the practical beginning of MLOps.

MLOps is a broad field. It can include feature stores, model registries,
automated retraining, deployment pipelines, monitoring, rollback, human
approval, and governance. This chapter stays narrower. It focuses on two ideas
that connect directly to this course:

```text
track the model training run
serve the trained model through a small API
```

The tool for the first part is MLflow. The tool for the second part is FastAPI.

## What MLOps Adds

Machine learning code often starts as an experiment:

```text
load data
train model
print metric
save file somewhere
```

That is fine for learning, but it breaks down quickly.

When more people, datasets, models, and training runs appear, the team needs a
better record. Otherwise, the same question keeps returning:

```text
Which run produced this model?
```

Good MLOps practice turns model training into an inspectable engineering
process.

A useful run record should contain:

- dataset name or location
- dataset version, snapshot, or manifest path
- training parameters
- model architecture or model family
- metrics over time
- evaluation plots
- model artifact
- example inputs and predictions
- code version or script name

This is the same pattern as the rest of the course. The exact tool changes, but
the goal is still repeatability.

## Experiment Tracking

Experiment tracking records what happened during model training.

Without tracking, two training runs may both print:

```text
validation accuracy: 0.82
```

But the number alone is not enough. You also need to know what produced it:

```text
run name: food-mobilenet-frozen
dataset: s3://preparation-for-ai-shared/datasets/chapter05/food101/manifest.csv
model: MobileNetV3 Small
selected labels: greek_salad, hamburger, pizza, ramen, sushi
epochs: 3
learning rate: 0.001
trainable backbone blocks: 0
final validation accuracy: 0.82
artifacts: confusion matrix, training curve, prediction examples, model bundle
```

That is much more useful. It lets another person compare, debug, reproduce, or
reject the result.

## MLflow

MLflow is an open-source tool for experiment tracking and model management.

In this workshop, MLflow is the shared tracking service. Training scripts send
evidence to MLflow while they run.

The basic MLflow concepts are:

- experiment: a named collection of related runs
- run: one execution of a training script
- parameter: an input choice, such as learning rate or number of epochs
- metric: a measured value, such as loss or validation accuracy
- artifact: a file, such as a plot, model file, or example image
- tag: searchable metadata, such as participant ID or dataset path

The code is deliberately simple:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow.mechatronics.lan")
mlflow.set_experiment("preparation-for-ai")

with mlflow.start_run(run_name="food-mobilenet-frozen"):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("validation_accuracy", 0.82, step=1)
    mlflow.log_artifact("confusion_matrix.png")
```

That small amount of code changes the training script from a local experiment
into a run that the whole team can inspect.

## Parameters, Metrics, And Artifacts

MLflow separates inputs, measurements, and files.

Parameters describe the choices made before or during training:

```text
epochs = 3
learning_rate = 0.001
batch_size = 32
trainable_backbone_blocks = 0
```

Metrics describe the result:

```text
training_loss at epoch 1
validation_loss at epoch 1
training_accuracy at epoch 1
validation_accuracy at epoch 1
training_loss at epoch 2
validation_loss at epoch 2
training_accuracy at epoch 2
validation_accuracy at epoch 2
final_validation_accuracy
```

Artifacts explain the result:

```text
training_curve.png
confusion_matrix.png
prediction_examples.png
model_bundle/model_state.pt
model_bundle/labels.json
model_bundle/model_config.json
```

This matters because a metric can be misleading on its own. A confusion matrix
may show that one class is never predicted correctly. Prediction examples may
show that the dataset contains ambiguous images. A model bundle may show which
labels the deployed service expects.

## Dataset Evidence Still Matters

MLflow does not replace the data platform.

The data evidence still comes from the earlier chapters:

- object storage keeps the assets
- manifests define multimedia datasets
- Parquet and Iceberg make analytical tables queryable
- Airflow can rerun workflows
- Superset can make prepared metrics visible

MLflow records which of those inputs were used by a training run.

For an image classifier, the important dataset evidence might be:

```text
manifest_key = datasets/chapter05/food101/manifest.csv
shared_bucket = preparation-for-ai-shared
selected_labels = greek_salad,hamburger,pizza,ramen,sushi
label_count = 5
train_rows = 3750
validation_rows = 1250
```

For an Iceberg table, the evidence might include a table name and snapshot ID.
For a manifest-backed multimedia dataset, the manifest path and label mapping
are often the most important starting point.

The rule is simple: a model run should point back to the dataset definition that
trained it.

## Comparing Runs

MLflow becomes useful when you have more than one run.

For example, you might train the same image classifier twice:

```text
run 1: freeze the pretrained image backbone
run 2: unfreeze the last backbone block
```

The second run may improve validation accuracy, but it may also train more
slowly or overfit faster. MLflow lets you compare the runs without searching
through local folders.

This is the practical workflow:

```text
change one training choice
run the script again
open MLflow
compare parameters, metrics, and artifacts
decide which run is the better candidate
```

That is a professional habit. The goal is not to chase the best possible model
in this workshop. The goal is to make model experiments visible and comparable.

## Model Artifacts

A trained model is a file, but the file alone is not enough.

A useful model artifact usually needs:

- model weights
- model architecture or model family
- preprocessing rules
- label mapping
- expected input shape
- example inputs
- evaluation evidence

In the lab, the model bundle contains:

```text
model_state.pt
labels.json
model_config.json
```

The API service can load that bundle because it contains both the weights and
the information needed to interpret the output classes.

This is the minimum handoff:

```text
training code
  -> model bundle
  -> API service
```

MLflow stores the same bundle as an artifact, so the model candidate is linked
to the training run that produced it.

## REST APIs

Training a model is not the same as using it.

One common deployment pattern is a REST API. Another program sends a request,
and the model service sends back a prediction.

The mental model is:

```text
client
  -> HTTP request
  -> /predict endpoint
  -> model service
  -> JSON response
```

For an image classifier, the request may contain an uploaded image. The response
may contain:

```json
{
  "predicted_label": "pizza",
  "confidence": 0.91,
  "model_name": "mobilenet_v3_small",
  "labels": ["apple_pie", "hamburger", "pizza"]
}
```

The API needs to do more than call `model(image)`.

It should:

- validate that the request contains an image
- apply the same preprocessing used during training
- load a known model bundle
- return a predictable JSON response
- expose a health endpoint
- expose model metadata

That is why deployment is software engineering, not only data science.

## FastAPI

FastAPI is a Python library for building APIs.

It is useful in this course because it makes the API visible. When you run a
FastAPI application, it automatically exposes interactive documentation:

```text
http://localhost:8008/docs
```

That page lets you call endpoints from a browser. Students can upload an image
to `/predict` and inspect the JSON response without writing a separate client.

The important code shape is:

```python
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(file: UploadFile):
    ...
```

FastAPI is not the only way to deploy a model, but it is concrete and easy to
inspect.

## What We Are Not Building

Production inference can become much more specialized.

Teams may use:

- Kubernetes deployments
- managed cloud model endpoints
- batch inference jobs
- streaming inference systems
- ONNX Runtime for portable model execution
- NVIDIA Triton Inference Server for high-throughput model serving
- edge deployments on industrial PCs, cameras, or gateways

Those tools matter, but they are not the main lab path here.

This chapter uses FastAPI because the deployment idea stays understandable:

```text
load model artifact
receive request
return prediction
```

That is enough to connect data preparation, model evidence, and model use.

## Monitoring Comes Next

A deployed model still needs monitoring.

At minimum, a team should ask:

- is the API healthy?
- how many requests succeed or fail?
- how long do predictions take?
- are input images similar to the training data?
- did the prediction distribution change?
- did quality degrade after the world changed?

Prometheus and Grafana are common tools for service metrics. MLflow can help
with model evidence. Business Intelligence dashboards can help inspect business
metrics. None of these tools solves the whole problem alone.

The practical lesson is that a model in production has two lives:

```text
the training run that produced it
the service behavior after deployment
```

Good MLOps keeps both visible.

## How The Labs Use These Ideas

The labs reuse the Food-101 manifest and the same five-label image-training
pattern from the multimedia chapter.

That is intentional. Choosing labels, caching images, and fine-tuning
MobileNet should already feel familiar. The new idea is what happens around the
model:

```text
train image classifier
  -> log run evidence to MLflow
  -> compare two runs
  -> choose a model bundle
  -> load that bundle in FastAPI
  -> call the model through a REST endpoint
```

The first lab focuses on evidence. You train a PyTorch model and inspect the
parameters, metrics, and artifacts in MLflow.

The second lab focuses on use. You start a small FastAPI service, call its
automatic API documentation, and then use a tiny frontend that calls the same
prediction endpoint.

Together, the labs show the minimum professional path:

```text
model training should leave evidence
model artifacts should be loadable
model predictions should be available through software
```

## Course Conclusion

The course started with local data inspection and small models.

It ends with a model training run that leaves evidence in MLflow and a small API
that can use the trained model.

The complete path is:

```text
raw data
  -> shared storage
  -> prepared layers
  -> queryable tables and manifests
  -> repeatable workflows
  -> dashboards
  -> tracked model runs
  -> model service
```

That is preparation for AI: not only training a model, but making the data and
model path understandable enough that another person can trust, rerun, compare,
and use it.
