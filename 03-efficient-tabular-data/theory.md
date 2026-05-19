# Chapter 3: Efficient Tabular Data

The previous chapters were about getting data into a reliable place and giving
it a useful structure. You started with local files, moved into object storage,
and then organized data into Bronze, Silver, and Gold layers.

This chapter narrows the focus to tabular data.

Tabular data is everywhere in AI projects: sensor measurements, event logs,
inspection records, transactions, user interactions, labels, feature tables,
validation results, and model metrics.

At small scale, a CSV file is often enough to get started. At platform scale,
CSV becomes a weak analytical format. It is text-based, has no strong schema,
does not compress as efficiently as columnar formats, and usually forces tools
to read much more data than the query actually needs.

The central idea in this chapter is:

```text
tabular data in object storage
  -> Parquet files
  -> query engine
  -> reusable outputs
```

The query engine may be local, such as Polars or DuckDB. It may also be shared,
such as Spark running on Kubernetes. The storage foundation stays the same:
prepared data lives in object storage, and compute engines read and write it.

## Object Storage Stays The Source Of Truth

The previous chapters matter here. You are not trying to build a better local
folder on one machine. You are building a small data platform where prepared
data lives in shared object storage.

A Gold dataset might be stored under a prefix like this:

```text
s3://preparation-for-ai-01/gold/forest-cover-type/features/
```

That prefix is more important than any one tool. It is the stable location where
prepared data can be read by different compute engines.

The same Gold dataset can be used by DuckDB for local SQL exploration, Polars
for Python dataframe work, Spark for shared distributed processing, Iceberg in
the next chapter, Airflow in a later workflow, Superset or another Business
Intelligence tool, and model training scripts.

That is the platform pattern:

```text
shared object storage
  -> multiple compute engines
  -> reusable datasets
```

This chapter is about making that object-storage dataset efficient enough to
query, transform, validate, and reuse.

## From Exchange Files To Analytical Files

CSV is an exchange format. It is useful because almost every tool can open it.
That does not make it a good storage format for analytical work.

A CSV file gives you lines of text:

```text
timestamp,machine_id,temperature,vibration,label
2026-05-14T08:00:00Z,M-001,72.4,3.1,normal
2026-05-14T08:01:00Z,M-001,72.9,3.4,normal
```

That is readable, but the file itself does not strongly describe the data. A
tool still has to infer types, parse strings, scan rows, and decide how to
handle missing or malformed values.

For Bronze data, that can be acceptable. Bronze is the raw landing layer.

For Silver and Gold data, you usually want something stronger:

- typed columns instead of guessed text values
- efficient compression
- fast reads of selected columns
- metadata that query engines can inspect
- broad support across Python, SQL engines, Spark, Business Intelligence tools, and cloud platforms

Parquet is the default answer for many modern data lakes because it gives you
those properties while still being a file format that fits object storage.

CSV and JSON do not disappear completely. They are still useful for raw
ingestion and exchange because they are easy to inspect, easy to produce, and
widely supported.

Compression is often useful for those raw text files. Formats such as
`.csv.gz` and `.json.gz` reduce storage cost and download time. The tradeoff is
that compressed text is still text: it is smaller, but it does not gain strong
schema information, column pruning, or Parquet-style partial reads. For repeated
analytical scans, Parquet is usually the better target format.

## Other Analytical And Scientific Formats

Parquet is not the only serious data format. Once you understand the difference
between row-oriented exchange files and columnar analytical files, it becomes
easier to place the alternatives.

Avro is a compact row-oriented serialization format with schemas. It often
appears in ingestion systems, event pipelines, and streaming-oriented
architectures. Avro is stronger than plain JSON because it carries schema
information, but it is not optimized for the same style of column pruning as
Parquet.

ORC is another columnar analytical format from the Hadoop and Hive ecosystem. It
has many of the same high-level goals as Parquet: typed columns, compression,
and efficient analytical reads. In many modern lakehouse projects, Parquet has
become the more common default, but ORC is still important to recognize.

Arrow IPC and Feather are useful for fast data exchange between dataframe tools
and languages. They are especially interesting when moving data between Python,
R, Rust, Java, and other runtimes without constantly converting everything back
to text.

HDF5 is common in scientific and engineering workflows, especially for
hierarchical arrays and large numerical datasets. Hyperspectral image cubes are
a good example: they are often better understood as chunked numerical arrays
than as tabular rows and columns.

Zarr has a similar scientific-array focus, but it is designed more directly for
chunked storage on cloud or object-storage systems.

Those formats can be excellent in the right context. In this course, Parquet is
the main tabular format because it is the most transferable lakehouse building
block: Polars, DuckDB, Spark, Iceberg, cloud warehouses, Business Intelligence
tools, and data platforms all understand it.

## Why Parquet Works Well For Analytics

Parquet files are not just compressed data blobs. They contain metadata.

The most important mental model is:

```text
Parquet file
  -> schema
  -> row groups
  -> column chunks
  -> statistics and metadata
```

You do not need to memorize the internals, but you should understand why they
matter. A query engine can inspect Parquet metadata before reading the full
data. It can often:

- understand column types without guessing
- read only selected columns
- skip row groups that cannot match a filter
- plan work before loading the data itself

That metadata becomes powerful because Parquet is columnar. CSV is
row-oriented: if you read a row, you read all fields in that row. Parquet stores
values from the same column together, which fits analytical queries that often
use only part of a dataset.

Example:

```sql
select
    machine_type,
    avg(vibration_mm_s) as avg_vibration
from measurements
where split = 'train'
group by machine_type;
```

This query does not need every column. It needs:

- `machine_type`
- `vibration_mm_s`
- `split`

If the table also contains timestamps, image paths, free-text notes, operator
IDs, location metadata, and dozens of sensor channels, Parquet lets the query
engine focus on the few columns that matter. That is the first practical
advantage: fewer bytes need to move.

The second advantage is compression. Values in one column often repeat or look
similar: product names, country codes, machine IDs, boolean flags, small integer
categories, or timestamps with regular spacing. Columnar storage gives
compression algorithms more structure to exploit. This is especially useful
with object storage because every unnecessary byte costs time, network traffic,
and money.

This is what makes Parquet work well with object storage: the query engine can
read the metadata first, then fetch only the pieces it needs.

The simplified comparison is:

```text
CSV:
  read text
  parse rows
  infer or apply types
  filter after reading

Parquet:
  read metadata
  select useful columns
  skip irrelevant row groups when possible
  decode only the required data
```

This difference is small in a tiny teaching dataset. It becomes important when
you have thousands of files, wide tables, or repeated queries from different
teams.

## File Layout And Partitioning

Once tabular data moves into object storage, a dataset is usually not one file.
It is a directory-like prefix that contains many files.

For example:

```text
s3://preparation-for-ai-01/silver/measurements/
  part-00000.parquet
  part-00001.parquet
  part-00002.parquet
```

That layout gives query engines room to work. Polars, DuckDB, and Spark can read
multiple Parquet files as one logical dataset. Spark can also split the work
over multiple executor pods.

Sometimes you organize those files into partitions:

```text
s3://preparation-for-ai-01/gold/features/
  split=train/part-00000.parquet
  split=train/part-00001.parquet
  split=test/part-00000.parquet
```

Partitioning is useful when queries often filter on the same stable columns:
date, customer, product line, machine type, location, or train/test split. If a
query only needs `split = 'test'`, the engine can avoid scanning the `train`
files.

Partitioning is not magic. If you partition by a column with too many unique
values, such as a unique event ID, you create too many tiny directories and too
many tiny files. That can be slower than having fewer larger files because object
storage has overhead for listing and opening objects.

The practical rule is simple: choose a layout that matches how the data is read.
Too few huge files limit parallelism. Too many tiny files create overhead. A good
layout gives query engines useful chunks of data without turning object storage
into a mess of microscopic files.

In this chapter, you only need the basic idea. In the next chapter, open table
formats add metadata on top of these files so that the platform can track table
snapshots, schema changes, and safer updates.

## A Query Engine Turns Files Into Something Database-Like

Parquet is not a database.

Parquet stores data efficiently, but it does not answer queries by itself. You
need a query engine.

The query engine is the software that:

- reads Parquet metadata
- chooses which columns and row groups to read
- applies filters
- performs joins and aggregations
- writes new outputs

That query engine can be local or shared.

Local engines run in your own course environment. Polars and DuckDB are the main
local query engines in this chapter. pandas with PyArrow sits nearby: it is
useful for local reading and inspection, but it is not the same kind of
optimizing engine as Polars or DuckDB.

Shared engines run on a platform. Spark, Trino, cloud warehouses, and managed
lakehouse engines fit that category.

The engineering question is not “which tool sounds most impressive?”. The
engineering question is:

```text
Where should this computation run?
```

For small or medium work, local execution is often faster, simpler, and easier
to debug. For large shared pipelines, distributed execution becomes useful.

## Python First: pandas And Its Limit

Most Python users meet tabular data through pandas.

That is a reasonable starting point. pandas is excellent for small and medium
dataframes, quick inspection, feature engineering experiments, plotting, and
many machine-learning workflows.

The limitation is that pandas is mostly an in-memory dataframe library. If you
write:

```python
import pandas as pd

measurements = pd.read_parquet("data/gold/features.parquet")
```

you usually materialize the dataframe in Python memory. That is fine for a
small prepared dataset. It becomes painful when the dataset is larger than the
memory available in your environment, or when you only needed three columns
from a very wide table.

pandas has practical mitigations. You can select fewer columns, use efficient
dtypes, read CSV files in chunks, and use Parquet through PyArrow. Those tricks
are useful, but they do not turn pandas into a distributed query engine.

This is why the chapter moves beyond plain pandas.

## Polars: Python Expressions Over Parquet

Polars is a fast dataframe library for Python. It is especially interesting
when you want dataframe code instead of SQL, but still want a query engine that
can optimize work before it runs.

The important Polars idea for this chapter is lazy scanning.

Instead of immediately loading a full file, you can build a query plan:

```python
import polars as pl

query = (
    pl.scan_parquet("data/gold/features/*.parquet")
    .filter(pl.col("split") == "train")
    .group_by("machine_type")
    .agg(
        rows=pl.len(),
        avg_vibration=pl.col("vibration_mm_s").mean(),
    )
    .sort("rows", descending=True)
)

result = query.collect()
print(result)
```

This looks like Python dataframe code, but it is still a query. Polars can
optimize lazy query plans before execution. For Parquet, that can include
reading only required columns and pushing filters closer to the scan.

Polars is also interesting for larger-than-memory workloads. With lazy queries
and streaming execution, some operations can be processed in chunks instead of
materializing the entire dataset at once. That does not make memory irrelevant,
and not every operation can stream equally well, but it gives you a much better
local path than "load the whole file into pandas, then process it".

Polars is still a local engine. It is not the same thing as running a
distributed Spark job. The useful step is that a single environment can already
do much more efficient analytical work before you need a cluster.

## SQL As A Shared Language, Not The Only Language

Python dataframe code is comfortable for data scientists. SQL is the historical
language of tabular data platforms.

SQL is not only for traditional databases. Modern data platforms use SQL over
Parquet files, object storage prefixes, lakehouse tables, cloud warehouses, Business Intelligence
datasets, and streaming or batch outputs.

SQL remains useful because it is readable across roles. A data engineer, data
scientist, analyst, and domain expert can often discuss the same query even if
they do not all write the surrounding Python code.

Example:

```sql
select
    machine_type,
    count(*) as rows,
    avg(vibration_mm_s) as avg_vibration,
    avg(label_failure) as failure_rate
from gold_features
where split = 'train'
group by machine_type
order by failure_rate desc;
```

The query tells a concrete story: use the training split, group rows by machine
type, count examples, calculate average vibration, calculate failure rate, and
sort the riskiest machine types first.

That readability matters when a transformation is naturally tabular and shared
between roles. SQL can make the logic easier to test, document, reuse, and
connect to Business Intelligence tools.

Python remains just as important. It is useful for parameters, orchestration
glue, custom parsing, validation code, model training, visualizations, and
application logic. Many professional data preparation steps combine both:

```text
object storage holds the input data
Python starts and configures the job
SQL expresses the transformation logic
Polars, DuckDB, or Spark executes the query
Parquet stores the output data
object storage keeps the result shared
```

## DuckDB: Local SQL Over Parquet

DuckDB is an embedded analytical database. You do not start a database server.
You import a library or open a command-line shell, then query files directly.

That makes it excellent for this stage of the workshop: it supports SQL, reads
Parquet directly, is fast for local analytical work, can write Parquet outputs,
and can also read from S3-compatible object storage.

The local version is simple:

```python
import duckdb

con = duckdb.connect()

result = con.sql("""
    select
        machine_type,
        count(*) as rows,
        avg(vibration_mm_s) as avg_vibration
    from read_parquet('data/gold/features/*.parquet')
    group by machine_type
    order by rows desc
""")

print(result)
```

The same idea can point at object storage when the environment is configured:

```sql
select
    machine_type,
    count(*) as rows
from read_parquet('s3://preparation-for-ai-01/gold/features/*.parquet')
group by machine_type;
```

DuckDB is not “less professional” because it runs locally. It is a serious tool
for inspection, analysis, validation, and small pipelines. DuckDB can also
process many larger-than-memory workloads by spilling intermediate data to disk.
That does not make memory irrelevant, but it makes DuckDB much more suitable
for analytical file processing than a naive "load everything into a dataframe"
approach.

Polars and DuckDB therefore occupy a similar place in this chapter:

```text
Polars: local Python dataframe query engine
DuckDB: local SQL query engine
```

You do not need both for every project. It is useful to recognize both because
professional teams often use a mixture of SQL-first and Python-first tools.

## When Local Compute Stops Being Enough

Polars and DuckDB are very strong local tools. For many projects, they are
enough for a long time.

Spark becomes useful when local execution is no longer the right shape: the data
is too large for one machine, the work is naturally parallel across many files
or partitions, the job should run on shared infrastructure, multiple teams
depend on the result, the same transformation should later be scheduled by
Airflow, or the platform already provides Spark as a standard compute layer.

Spark has more overhead than Polars or DuckDB. For tiny data, that overhead is
not worth it. For larger pipelines, the overhead buys you distributed execution
and platform integration.

Spark is not the only scale-out option. Trino is often used for distributed SQL
over data lakes. Dask and Ray are common Python-oriented distributed compute
frameworks. Apache Beam and Flink appear more often in dataflow and streaming
systems. Cloud platforms also provide their own serverless or managed query
engines.

In this course, Spark is the practical distributed engine because it is mature,
widely recognized in industry, and connects naturally to Parquet, Iceberg,
Airflow, and Kubernetes.

This is the scale-up story:

```text
Polars or DuckDB:
  one process
  one machine
  fast feedback

Spark:
  one job
  one driver
  one or more executors
  shared cluster compute
```

## The Minimal Spark Mental Model

Spark is a distributed compute engine. A Spark job has two important roles.

The driver coordinates the job. It reads your Spark application, builds the
execution plan, asks for resources, schedules work, and tracks progress.

The executors do the parallel work. They read data, apply transformations,
shuffle data when needed, and write outputs.

The simplified picture is:

```text
Spark application
  -> driver coordinates the job
  -> executors do the parallel work
  -> output is written back to object storage
```

That is enough Spark internals for this chapter.

There is a much deeper world underneath this: partitions, shuffles, caching,
memory tuning, adaptive query execution, file sizing, and cluster scheduling.
Those topics matter in production. They are not the goal here.

## Spark On Kubernetes In This Workshop

You will not administer Spark or Kubernetes in this course. You will use Spark
as a platform user.

In the lab, Kubernetes starts temporary pods for your Spark job. Those pods read
from object storage, process the data, and write results back to object storage.
When the job is finished, the running compute stops. The durable result is not
the pod. The durable result is the prepared dataset, report, or table written
back to object storage.

The simplified picture is:

```text
Spark job submitted to Kubernetes
  -> driver pod coordinates the job
  -> executor pods process partitions in parallel
  -> output is written back to object storage
  -> pods stop after the job finishes
```

The practical details of submitting and inspecting the Spark job belong in the
lab instructions. For the theory, the important idea is the separation of
concerns: object storage keeps the data durable, while Kubernetes provides
temporary shared compute when a local engine is no longer enough.

Spark and Kubernetes are both large topics. In this workshop, you only need the
platform-user view: submit a Spark job, inspect its status, and find the output
in object storage.

## Lab Preview

In the lab, you work with the forest cover type dataset again. The data is a
good fit for this chapter because it is genuinely tabular: one row describes one
land patch, the columns are measurements or categories, and the target is the
dominant forest cover type.

The shared bucket contains a Bronze CSV file. You first use Polars to scan that
CSV from object storage and write a partitioned Silver Parquet dataset to your
own bucket. That makes the file-format lesson concrete: the same rows become
more efficient analytical files.

Then you use DuckDB to query that same Silver Parquet dataset with SQL. DuckDB
creates a small Gold metric table grouped by cover type, then writes the result
back to object storage as Parquet.

Finally, you use Spark on the shared Kubernetes platform. Spark reads the same
Silver Parquet dataset, computes another metric table, and writes its result
back to object storage. The data stays in the lake; only the compute engine
changes.

The important lesson is interoperability. The Silver dataset is not "Polars
data". It is Parquet data in object storage. Polars can write it, DuckDB can
read it, Spark can process it with shared Kubernetes compute, and a later
Iceberg workflow can add table metadata on top of the same storage layer.

## Chapter Summary

This chapter is the tabular-data bridge between prepared layers and lakehouse
tables.

You now have the core pieces:

- Parquet stores tabular data efficiently in object storage.
- Polars lets you express local Python dataframe queries with lazy execution
  and support for many larger-than-memory workloads.
- SQL gives a shared language for tabular transformations.
- DuckDB lets you run local SQL directly over Parquet files.
- Spark gives the same preparation story a shared distributed-compute path.

The next chapter adds table state on top of Parquet files. That is where
snapshots, schema evolution, safer updates, and open table formats enter the
story.
