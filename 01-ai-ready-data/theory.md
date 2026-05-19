# Chapter 1: AI-Ready Data Starts Locally

AI does not start with a model. It starts with data.

That data can look very different from project to project:

- a small CSV file with customer records
- time series from sensors or machines
- text documents and support tickets
- audio recordings
- camera images
- video
- 3D point clouds
- hyperspectral images
- vendor-specific scientific files

Those are all "data", but they do not create the same engineering problem.
Some are small enough to inspect in a spreadsheet. Some are too large to move
around casually. Some fit naturally in tables. Others are files that need
metadata, labels, and specialized readers.

The first lesson is therefore simple:

```text
Different data types need different models, but all of them need reliable preparation.
```

## Expected Shape Of Data

Before you can prepare data, you need a basic expectation of what the data
should look like.

That expectation can be simple:

- which files should arrive?
- which columns should exist?
- which data types should those columns have?
- which values are physically or logically possible?
- which labels are required?
- which identifiers should be unique?
- who owns the source if something changes?

This is sometimes called a data contract. The word sounds formal, but the idea
is practical: downstream work should not silently break when an upstream system
changes.

In this chapter, a lightweight contract is enough. You should notice that "we
received a CSV" is weaker than "we received the expected CSV with the expected
columns, types, identifiers, and labels".

The simplest version can be written as plain text:

```text
covertype.csv
  required columns: elevation, slope, wilderness area, soil type, cover type
  cover type must be one of the documented target classes
  each row should describe one forest land patch

turbofan train csv
  required columns: engine, cycle, sensor measurements
  engine and cycle must identify a point in an engine trajectory
  every engine should have enough cycles to support a train/test strategy
```

A more mature version can become code:

```python
import pandas as pd

forest = pd.read_csv("01-ai-ready-data/covertype.csv")

required_columns = {"elevation", "slope", "wilderness_area", "soil_type", "cover_type"}
missing_columns = required_columns - set(forest.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

expected_labels = {0, 1, 2, 3, 4, 5, 6}
observed_labels = set(forest["cover_type"].dropna().unique())
unexpected_labels = observed_labels - expected_labels
if unexpected_labels:
    raise ValueError(f"Unexpected cover_type labels: {unexpected_labels}")
```

The first version is communication. The second version is automation. A good
data platform needs both.

In a professional team, these checks might be written in a data-quality
framework. In this first chapter, they can simply be Python code. The important
habit is that the expectation is no longer hidden in someone's head.

When a check fails, that is not automatically bad. Raw data can violate
expectations. The professional move is to make the violation visible before
model training.

## Data Types And Model Choices

The categories below are not strict boxes. Real projects often mix several
types of data, and the same dataset can be viewed in more than one way. The
point is to build a useful mental model: different data shapes create different
storage needs, cleaning problems, and model choices. Once you understand the
shape of the data, you can usually narrow down the first useful model family.
That does not mean the choice is automatic, but it gives you a sensible
starting point.

### Tabular Data

Tabular data is the familiar world of CSV exports, database tables, product
records, customer records, and machine measurements arranged in rows and
columns. It is often the best place to start because the feedback loop is
short: load a table, inspect columns, clean missing values, train a first
model.

Typical model tasks are regression, classification, and ranking. You often start
with classical supervised learning: linear models, logistic regression, decision
trees, random forests, gradient boosting, support vector machines, or
nearest-neighbor methods. Libraries such as scikit-learn make these models easy
to try quickly.

These models are not "old-fashioned" in a bad way. They are often the right
choice for production systems. They train quickly, they are easier to debug than
large neural networks, and they force you to understand the columns, target,
split, and metric. The hard part is often not the algorithm. It is agreeing on
what each column means, which values are allowed, which rows belong together,
and which features are safe to use.

### Time Series And Logs

Time series data adds order. Sensor streams, machine telemetry, energy
measurements, and transaction logs are not just "many rows". The timestamp is
part of the meaning.

This kind of data is used for forecasting, anomaly detection, predictive
maintenance, and event prediction. Preparation becomes more subtle because you
need to think about sampling rates, missing intervals, time alignment, and
train/test splits that do not accidentally leak the future into the past.

The first model question is whether you need a forecasting model, an anomaly
detector, an event predictor, or a normal supervised model with time-based
features. One common trick is to turn time into columns: previous values,
rolling averages, time since last event, hour of day, day of week, and similar
features. After that, a normal regression or classification model can already be
useful. More specialized statistical models and sequence models become
interesting when the time structure is the main part of the problem.

### Text And Documents

Text can be short and structured, like support tickets or product descriptions,
or long and messy, like reports, manuals, PDFs, and emails. The same file can
contain useful language, layout noise, personally identifiable information
(PII), confidential business information, and irrelevant boilerplate.

Text projects often involve classification, information extraction, embeddings,
retrieval augmented generation, or LLM evaluation. Before any of that works
well, you need to decide how documents are parsed, cleaned, split into chunks,
and connected back to their source.

Text can start simple. A bag-of-words representation plus a classical
classifier can already be useful for intent classification or document routing.
Once language nuance matters, embeddings and transformers become more
attractive. Large language models add another layer: they can classify,
summarize, extract, rewrite, and evaluate text, but they also make privacy, PII
handling, prompt design, and evaluation more important.

### Audio

Audio is a binary asset with time inside it. It might be speech, machine sound,
or environmental noise. A file is not enough by itself: you usually need the
sampling rate, duration, encoding, transcript, label, and sometimes the segment
inside the file that matters.

Common tasks include speech recognition, intent classification, sound
classification, and anomaly detection. The preparation problem is to keep the
audio bytes and the metadata connected.

Audio models often begin by turning the waveform into features that a model can
consume. Sometimes that means spectrogram-like representations. Sometimes it
means using a pretrained speech or audio model and fine-tuning it for a task.
For preparation, the model family is only half the story. Sampling rate,
duration, encoding, transcripts, labels, and segment boundaries can decide
whether the model input is meaningful.

### Images

Images are where file management becomes very visible. A dataset can contain
thousands or millions of JPEG, PNG, TIFF, microscopy, or inspection images.
Labels may live in folder names, CSV files, JSON annotations, or a labeling
tool.

Image classification is a common starting point. Object detection and
segmentation add richer annotations, such as bounding boxes and masks. Once
you move beyond small examples, GPU use, file size, label quality, and
annotation formats become central engineering concerns.

For classification, you can often start with a pretrained convolutional neural
network or vision transformer and fine-tune it on your own labels. That is much
more practical than training a large vision model from scratch. If the task is
not "which class is this image?", the model family changes. Object detection
needs bounding boxes. Segmentation needs masks. Those labels are more expensive
to create and maintain, so the preparation work becomes a large part of the
project.

### Video

Video combines images, time, compression, and very large files. Inspection
video, traffic footage, process recordings, and camera streams can quickly
become too heavy to copy around casually.

Model work can involve action recognition, object tracking, event detection,
or extracting frames for an image model. Preparation often means deciding which
clips matter, how frames are sampled, where annotations live, and how much
video you can afford to store and read during training.

A video model might work on sampled frames, short clips, or full temporal
sequences. That means the model might look like an image model applied to
frames, a video-classification model, or a more complex multimodal system. This
becomes expensive quickly. Video files are large, labels are costly, and
training can require serious GPU capacity. The preparation strategy often has
to reduce the problem before the model can solve it.

### Specialized Scientific And Spatial Data

There are many data types beyond the common tabular, text, audio, image, and
video examples. 3D point clouds, lidar scans, CAD-derived data, hyperspectral
images, medical volumes, and other scientific files all bring extra structure.
Sometimes that structure is geometry. Sometimes it is depth, a spectrum, a
calibrated measurement, or a specialized instrument format.

These datasets can support detection, segmentation, measurement, inspection,
classification, regression, anomaly detection, and material analysis. The model
might still be a classifier or regressor in the end, but the input pipeline is
not generic.

The preparation work depends heavily on metadata, calibration, normalization,
coordinate systems, units, reference frames, downsampling, and specialized file
readers. This is the broader lesson: once the data becomes more specialized,
data preparation is not just technical housekeeping. It becomes part of the
scientific or engineering meaning of the dataset.

In this workshop, you start small. A simple model is enough to learn the data
engineering ideas. You do not need the most advanced architecture to learn the
most important lesson: the model only deserves trust when the data path is
clear enough for another person to inspect.

## Train, Validation, And Test Splits

When you train a supervised model, you do not want to train and evaluate on the
same rows. A model can look impressive if it is tested on data it has already
seen. That does not prove it will work tomorrow.

That is why many datasets are divided into three parts.

The training split is the data the model is allowed to learn from. In code,
this is the data passed to `model.fit(...)`.

The validation split is used while you are still making decisions. You might
try different features, different model types, different hyperparameters, or
different cleaning rules. The validation score helps you choose between those
options.

The test split is kept aside for the end. It should answer a stricter question:
"After all our decisions, how well does this approach work on data we did not
use for training or tuning?"

In code, the shape is simple:

```python
model.fit(train[feature_columns], train[target])
validation_predictions = model.predict(validation[feature_columns])
test_predictions = model.predict(test[feature_columns])
```

The important habit is not the variable names. The important habit is separation
of evidence:

- train: learn the model
- validation: make modelling decisions
- test: estimate final performance

This only works if the split itself is meaningful. For a normal tabular
classification example, a stratified random split can be reasonable because it
keeps the class balance similar in train, validation, and test. For time series
or machine trajectories, a random row split can be misleading. If rows from the
same machine appear in both train and test, the model may partly recognize the
machine instead of learning a pattern that generalizes to new machines.

So when you see a metric, always ask: "What exactly was held out?"

## What Makes Data AI Ready?

An AI-ready dataset is not just a file that a model can read once. It is a
dataset that another person can understand, validate, and rebuild.

That usually means you keep several things together.

### Raw Source Data

Keep the original input somewhere durable. If a supplier sends a CSV, a machine
exports sensor logs, or a camera produces images, that original data is
evidence. You may clean it later, but you should not lose the starting point.

### Cleaned Records

Most models should not train directly on messy raw data. You often need a
cleaned version where column names are stable, units are consistent, invalid
values are handled, and records follow the expected shape.

The cleaned dataset should not be mysterious. You should be able to point to
the code that produced it.

### Labels And Targets

Supervised learning needs something to predict. That target might be a class
label, a numeric value, a future event, a bounding box, a transcript, or a mask.

Labels deserve special care because they often come from people or business
rules. If the labels are noisy, biased, or poorly defined, the model can learn
the wrong thing very confidently.

### Metadata

Metadata is data about the data. It can include source system, export time,
units, sensor type, image resolution, audio sampling rate, license, owner,
privacy classification, or known limitations.

Without metadata, the same bytes can become ambiguous. A column called
`temperature` is much less useful if you do not know the unit, sensor, and
context.

### Feature Definitions

A feature is an input variable used by the model. Some features are raw
columns. Others are derived, such as a rolling average, a time since last
maintenance, or a normalized measurement.

For AI-ready data, you should know how important features were calculated.
Otherwise, a model can depend on a column that nobody knows how to reproduce.

### Asset References

Not all training data fits in one table. Images, audio files, video clips,
PDFs, point clouds, and hyperspectral files are usually stored as separate
binary objects.

In that case, the dataset often contains references to those objects: file
paths, object-storage keys, checksums, labels, and metadata. The table does not
contain the image itself. It tells the training code where the image is and
what it means.

### Validation Results

Validation results are the checks that tell you whether the dataset looks
reasonable before training starts. Examples include row counts, missing-value
counts, allowed label values, schema checks, checksum checks, and range checks.

This is how you catch a broken export before it quietly becomes a broken
model.

### Model Metrics And Artifacts

After training, the model output also becomes evidence. Keep metrics, reports,
plots, model files, configuration, and links back to the dataset version that
produced them.

Strictly speaking, those are not the input dataset itself. They are part of the
evidence chain around the dataset. They help you answer the question: "What did
this data produce?"

The goal is simple: make the path from raw data to model result repeatable
enough that someone else can inspect it.

## Three Reliability Questions

When a team says a dataset is ready for AI, ask three questions.

### 1. Can We Explain Where It Came From?

Useful facts:

- source systems
- export time
- file names
- batch IDs or primary keys
- units
- known limitations
- who produced the data

This is not bureaucracy. It is how you avoid training on mysterious files that
nobody can explain later.

### 2. Can We Rebuild It?

Useful facts:

- exact raw inputs
- transformation code
- parameters
- train and test split rules
- validation checks
- output location

If the dataset cannot be rebuilt, the model result is hard to trust.

### 3. Can We Notice When It Changes?

Useful facts:

- row counts
- schema
- checksums
- dataset fingerprints
- validation metrics
- model metrics

AI systems are sensitive to quiet data changes. Good preparation makes those
changes visible.

## Data Size Changes The Architecture

A tabular dataset can be tiny. A folder of images can become large. Video,
audio archives, point clouds, and hyperspectral images can become heavy very
quickly.

As data grows, the way we work changes:

- local files are fine for first inspection
- shared storage becomes necessary for collaboration
- metadata becomes as important as the files themselves
- checksums and manifests help detect missing or changed assets
- query engines become useful when there are many files
- orchestration becomes useful when steps must run repeatedly
- experiment tracking becomes useful when many models are trained

That is the path of this workshop. We begin with local files because that is
the easy default: the data is on your own disk and the code is straightforward.
Then we move the same ideas into shared object storage and a small data
platform, because local disk is not enough once the work becomes collaborative,
larger, or repeatable.

Data size also changes how teams collaborate.

A CSV attached to an email can work for a first conversation. It is not a
serious platform. Once data is larger, more sensitive, or updated repeatedly,
the team needs shared storage, access control, metadata, and repeatable
processing.

That is why the first chapter is intentionally local and limited. It lets you
understand the shape of the problem before the course introduces the shared
architecture.

## The Small Data Platform

During the workshop, we build a small data platform workflow:

```text
local data inspection and small scikit-learn models
  -> object storage as the shared data-lake foundation
  -> Bronze, Silver, and Gold preparation layers
  -> Parquet, SQL, DuckDB, and Spark for tabular data
  -> open table formats for versioned lakehouse tables
  -> manifests and metadata for multimedia assets
  -> Airflow orchestration for repeatable workflows
  -> Business Intelligence dashboards for validation and communication
  -> MLOps evidence: metrics, models, APIs, and monitoring
```

In this course, we mostly use open-source tools that we can self-host on VIVES
infrastructure: S3-compatible object storage, DuckDB, Spark, Iceberg, Airflow,
Superset, and MLflow. That keeps the labs transparent. You can see the moving
parts without needing an account on a commercial cloud platform.

The ideas transfer directly to the big cloud providers. The product names
change, but the architecture is recognizable:

- on AWS, the same story might involve Amazon S3 for the data lake, AWS Glue
  or Amazon EMR for processing, Amazon Athena for SQL, Amazon MWAA for Airflow,
  Amazon QuickSight for dashboards, and Amazon SageMaker for machine learning
- on Azure, the same ideas map to Azure Blob Storage or Azure Data Lake
  Storage, Azure Data Factory, Azure Databricks or Azure Synapse, Power BI, and
  Azure Machine Learning
- on Google Cloud, the same ideas map to Cloud Storage, BigQuery, Dataproc or
  Dataflow, Cloud Composer for Airflow, Looker, and Vertex AI

You do not need to memorize those product names today. The important pattern is
more durable than the branding: shared object storage, repeatable processing,
queryable datasets, orchestration, validation, dashboards, and model evidence.

## Lab 1: Local First Contact

In this first lab, the work is local and Python-based.

You should:

- inspect raw CSV files
- understand what the target column means
- train small local models and inspect their plots
- compare simple and stronger feature sets
- notice when a metric looks better than the evaluation deserves

This is intentionally not a full pipeline yet. The platform starts when we move
the data into object storage, and the pipeline becomes more explicit when we
introduce Bronze, Silver, and Gold layers. First, you need to feel the problem
with your own hands.

## Lab Datasets

The lab uses real public datasets. They are still small enough for a workshop,
but they are not invented toy data.

The first dataset is tabular forest cover type data. Each row describes one
forest land patch, with cartographic measurements, wilderness area, soil type,
and a seven-class cover type label.

That gives you a classic first supervised-learning task:

```text
cartographic measurements -> forest cover type
```

This is close to many real tabular projects. You read a table, understand the
columns, check whether the target labels make sense, train a small classifier,
and ask whether the metric is believable. The seven target classes also make
the confusion matrix worth inspecting: some forest types are easier to separate
than others.

The second dataset is NASA turbofan degradation data. It also comes as a table,
but the meaning is different. Each engine has a sequence of cycles, and the
interesting question is how much useful life remains at each point in the
trajectory.

That gives you a time-series regression task:

```text
engine cycle history -> remaining useful life
```

This changes the machine-learning problem. A random row split can leak
information because the same engine appears in training, validation, and test
data. Keeping `engine` as a model input makes that problem worse: the model can
memorize an identifier instead of learning a reusable maintenance signal.

The lab first makes that mistake on purpose. The leaky model looks excellent,
but the split plot shows that rows from the same engine are scattered across
the splits. Then the honest version keeps whole engines together, removes the
engine identifier from the model inputs, and predicts the same
remaining-useful-life target under a more honest evaluation. The mean absolute
error becomes worse, but the evaluation becomes more believable.

That is the point of using both examples. The data type changes the preparation
strategy.

## Code Tools In The Labs

The Chapter 1 labs stay intentionally close to normal Python. You are not using
a large platform yet. You are learning the first habits that make the rest of
the platform useful.

You will use `pandas` to read CSV files, inspect columns, count missing values,
group rows, and prepare simple plots. In practice, that means most of the
first contact with a dataset starts with a dataframe:

```python
frame = pd.read_csv(path)
print(frame.columns)
print(frame.head())
```

You will use `scikit-learn` for small practical models. The point is not to
build the most accurate model possible. The point is to make the data problem
visible. In the forest cover lab, a classifier predicts the forest cover type.
In the turbofan lab, a regressor predicts remaining useful life.

The core loop is deliberately simple:

```python
model.fit(train[feature_columns], train[target])
predictions = model.predict(test[feature_columns])
```

Those tools are modest, but they already introduce a professional habit:
inspect the data, make assumptions explicit in code, train a first model, and
write down what happened.
