# Lab 8: Practical MLOps And Reproducibility

Start with [theory.md](theory.md), then review [slides.pdf](slides.pdf), then continue with the lab below.

In this lab you train an image classifier, log the run to MLflow, and serve the
trained model through a small FastAPI application.

You will use:

- PyTorch for model training
- MLflow for experiment tracking
- FastAPI for a local prediction API
- the Food-101 image manifest from object storage

The challenge is not to build the best food classifier. The challenge is to
make one trained model traceable: which data trained it, which parameters were
used, which metrics were produced, which artifacts explain the result, and how
another program can call it.

## Start In The Course Environment

Open a terminal in `/root/preparation-for-ai` and activate the shared Python
environment:

```bash
source .venv/bin/activate
```

Check that MLflow is configured:

```bash
grep '^MLFLOW_TRACKING_URI' participant.env
```

Then check that the server responds:

```bash
python3 - <<'PY'
import os
import requests
from dotenv import load_dotenv

load_dotenv("participant.env")
uri = os.environ["MLFLOW_TRACKING_URI"]
print("MLflow:", uri)
print("Health:", requests.get(f"{uri}/health", timeout=10).text)
PY
```

Open MLflow in a browser:

```text
http://mlflow.mechatronics.lan
```

The MLflow page has an experiment list on the left. This chapter uses:

```text
preparation-for-ai-chapter-08
```

## Lab 1: Track A PyTorch Training Run

Open:

```text
08-practical-mlops/lab1/track-image-model.py
```

The script fine-tunes MobileNet V3 Small on a small Food-101 image
classification task. You choose five labels, just like in Chapter 5. The
important new part is that the script also logs the training run to MLflow.

## Step 1: Train And Log A Model

You will find three TODOs:

- choose exactly five Food-101 labels
- give the MLflow run a short name
- fix the model call inside the PyTorch training loop

For the labels, you can reuse labels from Chapter 5 or open this image again:

```text
05-multimedia-assets/lab1/explore3_label_examples.png
```

For the run label, start with something descriptive:

```python
RUN_LABEL = "frozen-backbone"
```

The model-call TODO is the same core PyTorch idea you saw before:

```text
image batch -> model -> predictions -> loss -> optimizer step
```

After fixing the TODOs, run:

```bash
python3 08-practical-mlops/lab1/track-image-model.py
```

At the start, the script downloads and caches the selected images from object
storage. That can take a little while on the first run. The script prints cache
progress so you know it is still working. Once training starts, it prints
training loss, validation loss, training accuracy, and validation accuracy for
each epoch.

The script writes:

```text
08-practical-mlops/lab1/track1_training_curve.png
08-practical-mlops/lab1/track2_confusion_matrix.png
08-practical-mlops/lab1/track3_predictions.png
08-practical-mlops/lab1/api_sample_image.jpg
08-practical-mlops/lab1/model_bundle/
```

It also prints an MLflow run link.

First open:

```text
08-practical-mlops/lab1/track1_training_curve.png
```

This plot has four panels: training loss, validation loss, training accuracy,
and validation accuracy. Loss is the value the optimizer minimizes. Accuracy is
easier to read, but the validation panels are the ones that matter most because
they measure images the model did not train on.

Then open:

```text
08-practical-mlops/lab1/track2_confusion_matrix.png
```

Rows are true labels. Columns are predicted labels. Off-diagonal cells are
mistakes. If one label is often confused with another, that confusion should be
visible here.

Then open:

```text
08-practical-mlops/lab1/track3_predictions.png
```

This shows example validation predictions. Green prediction text means the
model was correct. Red prediction text means it was wrong. The interesting
question is whether the wrong examples are genuinely ambiguous.

Finally inspect:

```text
08-practical-mlops/lab1/model_bundle/
```

The bundle contains the deployable handoff:

- `model_state.pt`: learned model weights
- `labels.json`: output labels in the same order as the model logits
- `model_config.json`: model metadata and the MLflow run id

## Step 2: Inspect The Run In MLflow

Open MLflow:

```text
http://mlflow.mechatronics.lan
```

Click the `preparation-for-ai-chapter-08` experiment.

Find your run. The run name starts with your participant id and the label you
chose, for example:

```text
01-frozen-backbone
```

Click the run.

The run page has sections such as:

- `Metrics`
- `Parameters`
- `Artifacts`
- `Tags`

Look for these concrete pieces of evidence:

- `selected_labels`
- `learning_rate`
- `trainable_backbone_blocks`
- `training_loss`
- `validation_loss`
- `validation_accuracy`
- `dataset_manifest`
- `track1_training_curve.png`
- `track2_confusion_matrix.png`
- `track3_predictions.png`
- `model_bundle/`

This is the main lesson of Lab 1: the model result is no longer only a terminal
printout. It is a tracked run with data evidence, parameters, metrics, plots,
and a model bundle.

## Step 3: Run A Second Experiment

Go back to:

```text
08-practical-mlops/lab1/track-image-model.py
```

Change two things:

```python
RUN_LABEL = "unfreeze-one-block"
TRAINABLE_BACKBONE_BLOCKS = 1
```

Run the script again:

```bash
python3 08-practical-mlops/lab1/track-image-model.py
```

The second run trains slightly more of the pretrained neural network. It may
improve validation accuracy, or it may overfit faster. The point is that the
choice is now visible in MLflow.

In MLflow, open the experiment page and compare both runs.

Check:

- Which run has better final validation accuracy?
- Which run has lower validation loss?
- Did the extra trainable block help, or did it mostly improve training
  accuracy?
- Which run would you serve through the API?

You do not need to write a report. The MLflow comparison is the answer.

## Lab 2: Serve The Model With FastAPI

Lab 1 created:

```text
08-practical-mlops/lab1/model_bundle/
```

That bundle is the handoff from training to inference. Lab 2 loads it and
serves it through a local API.

## Step 4: Start The API

Open:

```text
08-practical-mlops/lab2/serve-image-model.py
```

Start the API:

```bash
python3 08-practical-mlops/lab2/serve-image-model.py
```

You should see Uvicorn print that it is running on port `8008`.

Keep that terminal open while you test the API. If you are using VS Code Remote
SSH, forward port `8008` so your browser can open the service.

## Step 5: Check The API From The Terminal

Open a second terminal in `/root/preparation-for-ai`, activate the environment
again, and run:

```bash
source .venv/bin/activate
```

Check the health endpoint:

```bash
curl http://localhost:8008/health
```

You should see:

```json
{"status":"ok"}
```

Check which model bundle is loaded:

```bash
curl http://localhost:8008/metadata
```

The response should include your labels and the MLflow run id.

Call the prediction endpoint:

```bash
curl -X POST \
  -F "file=@08-practical-mlops/lab1/api_sample_image.jpg" \
  http://localhost:8008/predict
```

The response is JSON. It should include:

- `predicted_label`
- `confidence`
- `model_name`
- `mlflow_run_id`

The `mlflow_run_id` is important. It connects the prediction service back to
the tracked training run.

## Step 6: Use The Browser API Docs

Open:

```text
http://localhost:8008/docs
```

FastAPI automatically creates this page from the Python endpoint definitions.

Try:

- `GET /health`
- `GET /metadata`
- `POST /predict`

For `/predict`, upload:

```text
08-practical-mlops/lab1/api_sample_image.jpg
```

This is a nice practical detail: the API is not only for Python code. The docs
page is an interactive client.

## Step 7: Use The Small Frontend

Open:

```text
http://localhost:8008
```

Upload the same sample image.

This page is only a tiny frontend. It does not contain the model. It calls the
same `/predict` endpoint that you called from `curl` and from the FastAPI docs.

That is the deployment pattern:

```text
frontend or client code
  -> REST API
  -> model bundle
  -> prediction response
```

## What You Should Notice

MLflow and FastAPI solve different problems.

MLflow answers:

```text
Which training run produced this model?
What data, parameters, metrics, and artifacts belong to that run?
```

FastAPI answers:

```text
Can another program send input to this model and receive a prediction?
```

The connection between both is the model bundle. Lab 1 creates it and logs it.
Lab 2 loads it and serves it.

That is the practical handoff at the end of the course:

```text
prepared data
  -> tracked training run
  -> model artifact
  -> prediction API
```
