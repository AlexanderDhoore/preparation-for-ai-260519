# Lab 3: Efficient Tabular Data

Start with [theory.md](theory.md), then review [slides.pdf](slides.pdf), then continue with the lab below.

This lab uses the forest cover type dataset again, but the focus is different
from Chapter 1.

In Chapter 1, you used the dataset to explore a machine-learning problem. In
this chapter, you use the same kind of tabular data to practice efficient
storage and query patterns:

- Polars converts Bronze CSV into Silver Parquet.
- DuckDB reads the same Parquet data with SQL and writes a Gold metric table.
- Spark reads the same Parquet data through shared Kubernetes compute.

The important idea is interoperability. The dataset lives in object storage.
The compute engine can change.

## S3 Layout

These shared Bronze objects are already available:

```text
s3://preparation-for-ai-shared/bronze/chapter03/covertype.csv
```

During the lab, you write these objects to your own participant bucket:

```text
s3://preparation-for-ai-XX/silver/chapter03/covertype/
s3://preparation-for-ai-XX/gold/chapter03/duckdb_metrics.parquet
s3://preparation-for-ai-XX/gold/chapter03/spark-metrics/
```

Replace `preparation-for-ai-XX` with your own participant bucket.

The cover type labels are the same as in Chapter 1:

- `0`: Spruce/Fir
- `1`: Lodgepole Pine
- `2`: Ponderosa Pine
- `3`: Cottonwood/Willow
- `4`: Aspen
- `5`: Douglas-fir
- `6`: Krummholz

## Start In The Course Environment

Run:

```bash
source .venv/bin/activate
```

These labs use the `participant.env` file that was introduced in Chapter 2.
The scripts load it automatically when they connect to object storage.

## Step 1: Polars Converts CSV To Parquet

Open:

```text
03-efficient-tabular-data/lab1/polars-parquet.py
```

This script starts from a Bronze CSV file in the shared bucket. CSV is easy to
exchange, but it is not a great repeated analytics format. Your job is to turn
it into a Silver Parquet dataset in your own bucket.

Complete the TODOs:

- `SOURCE_CSV_KEY`: use the shared Bronze CSV key from the S3 layout above.
- `PARTITION_COLUMN`: use `wilderness_area`.

Then run:

```bash
python3 03-efficient-tabular-data/lab1/polars-parquet.py
```

Expected local outputs:

```text
03-efficient-tabular-data/lab1/polars1_storage_summary.png
03-efficient-tabular-data/lab1/polars2_rows_by_wilderness_area.png
03-efficient-tabular-data/lab1/polars3_class_balance.png
```

Expected S3 output:

```text
s3://preparation-for-ai-XX/silver/chapter03/covertype/wilderness_area=0/00000000.parquet
s3://preparation-for-ai-XX/silver/chapter03/covertype/wilderness_area=1/00000000.parquet
s3://preparation-for-ai-XX/silver/chapter03/covertype/wilderness_area=2/00000000.parquet
s3://preparation-for-ai-XX/silver/chapter03/covertype/wilderness_area=3/00000000.parquet
```

The exact Parquet filenames may vary between library versions, but the
important structure is the `wilderness_area=...` folders. That is the partition
layout.

First open:

```text
03-efficient-tabular-data/lab1/polars1_storage_summary.png
```

This plot compares the Bronze CSV size with the Silver Parquet size. It also
shows how many Parquet objects Polars wrote. The point is not only that Parquet
is smaller. The more important shift is that you now have a typed analytical
dataset, partitioned by `wilderness_area`, instead of one exchange-style CSV
file.

Then open:

```text
03-efficient-tabular-data/lab1/polars2_rows_by_wilderness_area.png
```

This plot shows the partition layout you created. Each bar corresponds to a
`wilderness_area` partition in object storage. The partitions are not equally
large, so the files will not contain equal amounts of data.

That is not automatically a problem. Partitioning is useful when later queries
often filter on the partition column. If many queries ask for one
`wilderness_area`, the query engine can skip the other partition folders instead
of scanning every Parquet file. You can think of this as a simple storage-level
index: useful for the right filter, but not a magic speedup for every query.

Then open:

```text
03-efficient-tabular-data/lab1/polars3_class_balance.png
```

This plot shows the forest cover type balance overall and inside each
`wilderness_area`. The first panel uses all rows. The next four panels show one
partition at a time.

Questions:

- Which cover types appear almost everywhere?
- Which cover types are concentrated in only one or two wilderness areas?
- Why would this matter if a later model is trained only on one partition?

Before moving on, check the S3 prefix that the script printed. The next step
depends on that exact Silver Parquet dataset. If the Parquet files are missing,
DuckDB has nothing to query.

You will copy that Silver prefix into the DuckDB script next.

## Step 2: DuckDB Queries The Same Parquet Data

Open:

```text
03-efficient-tabular-data/lab1/duckdb-sql.py
```

This script reads the Silver Parquet dataset that Step 1 wrote. There is no
Polars dependency in this step. DuckDB reads the same files directly from
object storage.

Complete the TODOs:

- `SILVER_PREFIX`: use the Silver Parquet prefix from Step 1:
  `silver/chapter03/covertype`.
- in `summary_sql`: replace `TODO_FILL_SQL_ROW_COUNT_EXPRESSION` with the SQL
  expression that counts rows per group. The hint in the script is deliberately
  close to the answer.

The grouping and measurement columns are already chosen in the script:
`cover_type` and `elevation`. Your job is to read the SQL query and complete
the small aggregate expression that makes the metric table work.

Then run:

```bash
python3 03-efficient-tabular-data/lab1/duckdb-sql.py
```

Expected local outputs:

```text
03-efficient-tabular-data/lab1/duckdb1_metric_table.csv
03-efficient-tabular-data/lab1/duckdb2_metric_graphs.png
```

Expected S3 output:

```text
s3://preparation-for-ai-XX/gold/chapter03/duckdb_metrics.parquet
```

First open:

```text
03-efficient-tabular-data/lab1/duckdb1_metric_table.csv
```

This is a readable local copy of the Gold metric table DuckDB wrote to object
storage as Parquet. Each row is one forest cover type. The columns contain row
counts, the number of wilderness areas and soil types represented, and minimum,
average, and maximum `elevation`.

Questions:

- Which cover type has the most rows?
- Which cover type has the highest average elevation?
- Which cover types appear in fewer wilderness areas?

Then open:

```text
03-efficient-tabular-data/lab1/duckdb2_metric_graphs.png
```

This plot visualizes every numeric column from the metric table:

- rows per cover type
- number of wilderness areas per cover type
- number of soil types per cover type
- average elevation per cover type
- minimum elevation per cover type
- maximum elevation per cover type

That gives you a quick visual check before you trust the Gold table. The
pattern should be clear: different cover types live in different parts of the
landscape.

This is the main point of the first two labs:

```text
Polars wrote Parquet.
DuckDB read the same Parquet.
The data stayed in object storage.
```

Do one quick investigation before continuing: compare the group column in the
SQL query with the partition column from Step 1. They do not have to be the
same.

Step 1 partitioned the files by `wilderness_area`. That is a storage-layout
choice. It decides which folders and Parquet files exist in object storage, and
it can make queries faster when they filter on `wilderness_area`.

Step 2 groups the data by `cover_type`. That is an analysis choice. It decides
which question the metric table answers. A query can group by a different
column than the partition column; the partition column is about where bytes
live, while the group column is about what comparison you want to make.

## Step 3: Spark Reads The Same Parquet Data

Spark is the shared-compute version of the same pattern. You do not administer
Spark or Kubernetes here. You generate a Kubernetes `SparkApplication`, submit
it, inspect it, and check the output in object storage.

First, open:

```text
03-efficient-tabular-data/lab1/spark-application.py
```

Complete the TODO:

- `SPARK_OUTPUT_PREFIX`: use `gold/chapter03/spark-metrics`.

Then generate the Kubernetes manifest:

```bash
python3 03-efficient-tabular-data/lab1/spark-application.py
```

Open the generated manifest before applying it:

```text
03-efficient-tabular-data/lab1/spark1_application.yaml
```

This is the Kubernetes object you are about to submit. It contains:

- a `ConfigMap` with the PySpark script
- a `SparkApplication` that runs that script on Kubernetes

Look for `driverArgs` in the YAML. That is where this lab passes arguments into
the PySpark script. The first value should point to the Silver Parquet dataset
created in Step 1. The second value should point to the Spark Gold metrics
prefix in your participant bucket.

At a high level, this is what is happening:

```text
you apply spark1_application.yaml
        |
        v
Kubernetes receives a SparkApplication object
        |
        v
Spark Operator creates Spark pods
        |
        v
driver pod coordinates the job
executor pod reads Parquet and writes metrics
```

Kubernetes is the platform that starts and supervises containers. Spark is the
distributed query engine running inside those containers. The Spark Operator is
the small Kubernetes add-on that understands `SparkApplication` objects and
turns them into driver and executor pods. You do not need to manage those pods
manually in this course, but it is useful to recognize what you are seeing.

`participant.kubeconfig` is the credential file that lets your course
environment submit Spark jobs to the shared Kubernetes platform.

Submit the Spark job:

```bash
KUBECONFIG=participant.kubeconfig kubectl apply -f 03-efficient-tabular-data/lab1/spark1_application.yaml
```

Inspect the Spark application:

```bash
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX get sparkapp
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX describe sparkapp chapter03-spark
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX get pods
```

The Spark application should move through states such as `Submitted`,
`DriverRequested`, `RunningHealthy`, and `Succeeded`. Because the driver pod is
kept around for inspection, the final visible state may become
`TerminatedWithoutReleaseResources`. That is okay when the transition history
contains `Succeeded`.

Watch the driver logs:

```bash
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX get pods
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX logs <driver-pod-name> --tail=100
```

Expected S3 output:

```text
s3://preparation-for-ai-XX/gold/chapter03/spark-metrics/
```

The Spark job also prints the metric table to the driver logs. Look for this
line:

```text
Spark metric table:
```

Then compare the logged table with `duckdb1_metric_table.csv` from Step 2. The
numbers should describe the same source dataset, but the execution model is
different:

```text
DuckDB: local SQL engine in your participant environment
Spark: shared compute job running on Kubernetes
```

When you are done inspecting it, clean up your Spark application:

```bash
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX delete sparkapp chapter03-spark
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX delete configmap chapter03-spark-code
```

## Step 4: Check When Spark Is Actually Needed

Spark is useful because it can distribute work over many machines. But do not
reach for Spark only because a dataset feels large. DuckDB and Polars can go
much further than many people expect on one strong machine.

Open:

```text
https://yourdatafitsinram.net/
```

Try a few memory sizes. The useful question is:

```text
Would one large machine be simpler, or do I really need distributed compute?
```

## What You Learned

You used one dataset through three query styles:

- Python dataframe expressions with Polars
- SQL with DuckDB
- shared Kubernetes compute with Spark

The storage pattern stayed stable. Bronze CSV became Silver Parquet. DuckDB and
Spark both read that Parquet without needing a new copy of the dataset in a
different system. That is the practical value of efficient tabular files in
object storage.

The important question to carry forward is:

```text
Which parts of my workflow depend on the storage format, and which parts can
change query engine without changing the data?
```
