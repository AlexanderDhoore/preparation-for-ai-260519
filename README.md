# Preparation For AI: From Raw Data To Reliable Models

Welcome to the Preparation For AI course.

These labs take you from first contact with raw data to a small data platform
that can support reliable analytics and machine learning. You will work with
real public datasets, Python scripts, object storage, prepared data layers,
query tools, orchestration, dashboards, and practical MLOps evidence.

Each chapter contains two things:

- `theory.md` with the written course theory
- `README.md` with the hands-on lab instructions

Start with the chapter README, run the commands, inspect the code, fix the
small TODOs, and look at the generated outputs. The goal is not only to make
the scripts run. The goal is to understand what the data platform is doing.

## Start Here

After you are connected to your prepared Linux environment, create a Python
environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then begin with Chapter 1:

```text
01-ai-ready-data/README.md
```

Some labs ask you to complete small TODOs, inspect code, change configuration,
or explore a prepared service. Follow the chapter instructions and pay
attention to the outputs, errors, dashboards, or files the lab asks you to
inspect.

## Lab Overview

### 1. [AI-Ready Data Starts Locally](01-ai-ready-data/)

Start locally with tabular and time-series datasets. You inspect raw files,
define the expected shape of data, train small scikit-learn models, and see why a
model metric is only useful when the data path is understandable.

### 2. [Object Storage And Data Lake Layers](02-object-storage/)

Move from local files to S3-compatible object storage. You work with audio
objects, read manifests, and write your own Silver and Gold outputs back to a
personal bucket. You also introduce the Bronze, Silver, and Gold vocabulary for
organizing a data lake into raw, cleaned, and use-case-ready layers.

### 3. [Efficient Tabular Data](03-efficient-tabular-data/)

Focus on tabular data. You use Parquet, SQL, Polars, DuckDB, and Spark to query and
transform prepared datasets more efficiently than plain CSV files.

### 4. [Open Table Formats](04-open-table-formats/)

Go beyond folders of Parquet files. You learn why table metadata, snapshots,
schema evolution, and formats such as Apache Iceberg matter in a lakehouse. You
use PyIceberg, DuckDB, and Spark against the same table.

### 5. [Multimedia Assets For Machine Learning](05-multimedia-assets/)

Handle non-tabular data such as images, text documents, audio, video, and
scientific files. You use manifests to connect large objects in storage with
labels, splits, metadata, and neural-network training code.

### 6. [Airflow, Orchestration, And Repeatability](06-orchestration/)

Move from manual scripts to repeatable workflows. You use Airflow-style
orchestration concepts: DAGs, tasks, schedules, triggers, retries, logs, and
validation gates.

### 7. [Business Intelligence](07-business-intelligence/)

Make prepared analytical data understandable to a wider team. You use Superset
and Trino to query an Iceberg table, build charts, and assemble a dashboard.

### 8. [Practical MLOps And Reproducibility](08-practical-mlops/)

Close the loop from prepared data to model training evidence. You explore
MLflow tracking, model artifacts, and a small FastAPI prediction service.

## Goal Of The Course

By the end of these labs, you should be able to explain how raw data becomes
reliable input for analytics and machine learning:

- inspect and validate raw data
- store data in a shared object-storage layer
- organize data lake outputs with Bronze, Silver, and Gold layers
- use Parquet, SQL, Spark, and table metadata for tabular data
- manage multimedia assets with manifests and metadata
- automate repeatable steps with orchestration
- make validation and metrics visible in dashboards
- connect prepared datasets to model training and MLOps evidence

## Prerequisites

- Basic Python knowledge
- Basic terminal usage
- Willingness to read and edit small Python scripts
- Some familiarity with data science or machine learning concepts
- Visual Studio Code with Remote SSH, if you are using a prepared remote
  environment
