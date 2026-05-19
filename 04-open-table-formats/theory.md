# Chapter 4: Open Table Formats

In Chapter 3 you stored an efficient tabular dataset as Parquet files in object
storage. That is already a strong pattern. The files are open, compressed, and
queryable. Polars can write them, DuckDB can query them, and Spark can process
them on shared compute.

But a folder of Parquet files is still only a folder of files.

That becomes a problem when the dataset changes over time. New files arrive. A
column is added. A bad batch is corrected. A model is trained on Monday and
another model is trained on Friday. At that point, you need a stronger answer
than "the data was somewhere under this S3 prefix".

An open table format adds table metadata on top of files in object storage. The
data can still live as Parquet, but the table format records which files belong
to the table, which schema is active, which snapshots exist, and how the table
changed.

The main examples are Apache Iceberg, Delta Lake, and Apache Hudi. In this
chapter you use Apache Iceberg as the concrete example.

## The Problem With Plain Parquet Folders

A Parquet dataset in object storage might look like this:

```text
s3://preparation-for-ai-shared/bronze/chapter04/nyc-taxi/
  yellow_taxi_2024-01_sample.parquet
  yellow_taxi_2024-02_sample.parquet
```

That layout is useful. The files are compressed and columnar. Each file has a
clear month in the filename, and query engines can read the Parquet metadata
before scanning the data. In the lab, these are instructor-prepared random
samples from the public NYC taxi Parquet files, still large enough to behave
like real analytical data.

The problem is that the folder itself does not tell you the full table story.

It does not tell you:

- which files are part of the current table
- which files are old and should be ignored
- which schema belongs to which version
- whether a write completed successfully
- what changed between yesterday's data and today's data
- which dataset version trained a model

You can solve some of that with naming conventions and manifests. Those are
important, especially for multimedia assets later. But when structured tables
become central to a platform, you usually want the table itself to carry this
state.

That is where open table formats come in.

## Why Iceberg Is Not Just More Parquet

Iceberg commonly stores table data as Parquet, but Iceberg is not "a different
Parquet file".

Parquet is the file format. It controls how rows and columns are encoded inside
files.

Iceberg is the table format. It controls how data files become a table over
time.

The catalog is the naming and coordination layer. It lets multiple engines use
the same table name and agree on the current metadata.

The distinction matters:

- Parquet makes scans efficient.
- Iceberg makes table state explicit.
- The catalog makes table names usable across tools.

In Chapter 3, different engines could read the same Parquet files because the
files were open and standard.

In Chapter 4, different engines can agree on the same table because the metadata
and catalog protocol are open and standard.

## What An Open Table Format Adds

An open table format does not replace object storage. It adds metadata around
the files in object storage.

For an Iceberg table, the physical storage still contains ordinary data files
and metadata files:

```text
chapter04/iceberg-warehouse/
  preparation_for_ai_01/
    yellow_taxi_trips/
      data/
        ...
      metadata/
        00000-...metadata.json
        snap-...avro
        ...manifest...
```

You do not need to memorize the exact filenames, but the shape is worth
understanding. Iceberg tables are built from a few concrete pieces.

Data files store rows. In this course, those data files are usually Parquet.

Metadata JSON files describe the table. They record the schema, partition
specification, table properties, snapshots, and the current table state.

Manifest files describe groups of data files. They can record file paths,
partition values, row counts, and useful statistics. This lets a query engine
plan a scan without blindly listing every object under a prefix.

The catalog points tools to the current table metadata. Without that pointer,
every engine would need to be told exactly where the latest metadata file is
stored.

When a write happens, Iceberg does not edit one Parquet file in place. It writes
new files and commits new metadata. That new metadata becomes a new table state.
This is closer to how databases behave, but the bytes still live in object
storage.

A simplified first write looks like this:

```text
catalog table name
  -> metadata/00000-metadata.json
       -> snapshot A
            -> manifest list
                 -> manifest file
                      -> data/00000.parquet
                      -> data/00001.parquet
```

After an append, Iceberg can create a new metadata file and a new snapshot:

```text
catalog table name
  -> metadata/00001-metadata.json
       -> snapshot B
            -> previous data files
            -> newly appended data files
```

The older metadata and data files do not have to disappear immediately. That is
what makes time travel, rollback, and reproducible training references possible.
Maintenance jobs can clean up old files later when the team decides they are no
longer needed.

## Open Tables Keep The Shared Storage Idea

In Chapter 3, Polars, DuckDB, and Spark could all work with the same Parquet
files because the data lived in object storage and the file format was open.
That was already powerful: the storage layer was shared, but the compute layer
could change.

Open table formats keep that idea, but add a stronger table layer on top.

The data still lives in object storage. The data files can still be open formats
such as Parquet. Different tools can still connect to the same dataset. The
difference is that they no longer have to guess which files form the current
table. They can use the catalog and table metadata.

That gives you database-like behavior without moving the data back into a
traditional database:

- named tables instead of only paths
- snapshots instead of only overwritten folders
- schema evolution instead of accidental file differences
- appends, overwrites, deletes, and updates as table operations
- ACID-style commits on top of object storage
- metadata that query engines can use for planning
- a common table state that different engines can agree on

ACID stands for atomicity, consistency, isolation, and durability. In a database
course, those words get a very precise meaning. Here, the practical idea is
simple: a table write should either become a valid new table state, or it should
not become visible as a successful table update.

That matters because object storage itself does not behave like PostgreSQL. You
cannot safely update one row inside one object as if the object were a database
page. Table formats solve this differently. Engines may write replacement data
files, write delete files that describe removed rows, and write metadata files,
then commit a new table state. A delete or update may therefore create new
files behind the scenes. The user sees a table operation, but the storage layer
still stores immutable objects and metadata.

The last point is the most important one for this chapter. An open table format
does not mean "one tool owns the data now". It means the table has a shared,
open description that several tools can understand.

A Python library can create or inspect a table. A SQL engine can query it. A
distributed engine can process it at larger scale. The object store remains the
durable storage layer, and the catalog keeps track of table identity and current
metadata.

That is the lakehouse pattern in miniature: object storage keeps the flexibility
of a data lake, while the table format adds enough structure to make the data
feel more like a managed analytical table.

## Catalogs And Polaris

A catalog is the service that lets tools find tables by name.

Without a catalog, you often point tools directly at paths:

```text
s3://preparation-for-ai-01/chapter04/iceberg-warehouse/.../metadata/00000-...metadata.json
```

That is not how most people want to work.

With a catalog, you use a stable table name:

```sql
select *
from course.preparation_for_ai_01.yellow_taxi_trips;
```

The catalog resolves that name to the current table metadata. Spark, DuckDB,
Trino, Flink, Dremio, Snowflake, or another engine can then use the table
without every user copying object-storage paths around.

The catalog is also where table permissions can live. In this workshop, each
participant gets a separate Iceberg namespace such as `preparation_for_ai_01`
or `preparation_for_ai_02`. That is almost the same as the S3 bucket and
Kubernetes namespace, but not exactly. Buckets and Kubernetes namespaces use
DNS-style names with dashes. Iceberg namespaces are used inside SQL, so they
use underscores.

For this chapter, the shared catalog service is Apache Polaris.

Polaris is not the storage layer. Ceph RGW remains the object-storage layer.
Polaris is the catalog service that tracks Iceberg table names, namespaces,
snapshots, schemas, and permissions.

The workshop platform uses one shared Polaris service. That is intentional. In a
real company, you usually do not run a separate catalog for every user. You run
one platform service and give each team or user access to the parts they own.

The simplified workshop shape is:

```text
participant VM
  -> PyIceberg, DuckDB, or kubectl command
  -> shared Polaris REST catalog
  -> participant namespace in the catalog
  -> participant S3 bucket in Ceph RGW
```

Polaris gives the table a shared name. Ceph stores the bytes. PyIceberg,
DuckDB, and Spark are engines or client libraries that connect to the catalog.

## Snapshots

A snapshot is a recorded table state.

When you create an Iceberg table, the table gets an initial snapshot. When you
append data, overwrite data, or replace the table, Iceberg can create another
snapshot. Each snapshot points to the manifests and files that were valid at
that moment.

This matters because teams need to ask historical questions:

- Which table state was used for last week's training run?
- Did the schema change before or after the model was trained?
- Did a bad batch of labels enter the table?
- Can you inspect the previous version without copying the entire dataset?

In Spark, Iceberg exposes snapshots through metadata tables. A normal table may
be called:

```sql
course.preparation_for_ai_01.yellow_taxi_trips
```

The snapshots for that table can be queried as:

```sql
select *
from course.preparation_for_ai_01.yellow_taxi_trips.snapshots;
```

That is a useful mental model. The table is not only rows. The table also has
metadata you can query.

## Schema Evolution

Data schemas change. That is normal.

You might add a column because you calculated a new derived feature. You might
rename a field because the old name was confusing. You might change a type
because the raw source system changed. You might remove a column because it
contained bad or private information.

Without a table format, schema evolution often becomes a set of fragile
assumptions in code. One file has a new column, another file does not, and a
third job silently fills missing values with nulls. Sometimes that is fine.
Sometimes it breaks a training pipeline in a way that is hard to diagnose.

Open table formats make schema change explicit. They record schema information
in table metadata and let compatible engines understand how the table evolved.

That does not mean every schema change is safe.

Adding a nullable column is usually easier to handle than changing the meaning
of an existing label. Renaming an identifier can break joins. Changing a target
column can make old model metrics incomparable. Dropping a feature can break a
training script that still expects it.

The table format gives you a technical mechanism. You still need engineering
discipline.

In practice, a schema change becomes a table operation instead of an
undocumented file accident. For example, Spark SQL can add a column to the table
metadata:

```sql
alter table course.preparation_for_ai_01.yellow_taxi_trips
add column trip_minutes double;
```

New writes can populate that column. Older snapshots can still be interpreted
with the schema that existed at the time. That is the practical difference
between "some Parquet files under a prefix changed" and "the table evolved in a
recorded way".

## Partition Evolution

Open table formats also help with partition evolution.

In Chapter 3, the partition layout was visible in the object keys:

```text
split=train/
split=test/
```

That is a useful file layout, but it couples readers to the folder structure.
In a taxi table, you might start with one file per month and later partition by
pickup date or pickup location. If you change that layout manually, older files
and newer files may no longer look the same.

Iceberg can track partition information in metadata. That means the table can
evolve its partitioning rules without forcing every reader to understand every
historical folder layout manually.

You should not interpret this as magic. Bad partition choices can still hurt
performance. Tiny files can still be a problem. Tables still need maintenance.
But the table format gives the platform a cleaner place to record and manage
those decisions.

## Operations You Should Recognize

You do not need to become an Iceberg administrator in this course. You should
recognize the basic operations.

Create a namespace from Python:

```python
catalog = load_catalog("course", type="rest", uri="https://polaris.example/api/catalog")

if not catalog.namespace_exists("preparation_for_ai_01"):
    catalog.create_namespace(
        "preparation_for_ai_01",
        properties={"location": "s3://preparation-for-ai-01/chapter04/iceberg-warehouse/preparation_for_ai_01"},
    )
```

Create a table and append Arrow data:

```python
table = catalog.create_table(
    "preparation_for_ai_01.yellow_taxi_trips",
    schema=arrow_table.schema,
    location="s3://preparation-for-ai-01/chapter04/iceberg-warehouse/preparation_for_ai_01/yellow_taxi_trips",
)

table.append(arrow_table)
```

Query a table:

```sql
select payment_type, count(*)
from course.preparation_for_ai_01.yellow_taxi_trips
group by payment_type;
```

Inspect snapshots:

```sql
select committed_at, snapshot_id, operation
from course.preparation_for_ai_01.yellow_taxi_trips.snapshots
order by committed_at;
```

Add a column from Spark SQL:

```sql
alter table course.preparation_for_ai_01.yellow_taxi_trips
add column trip_minutes double;
```

Append a new batch:

```python
february_trips.writeTo("course.preparation_for_ai_01.yellow_taxi_trips").append()
```

The exact syntax depends on the engine. The main point is that these operations
update table metadata in a controlled way.

The lab uses the append operation because it is the clearest first example:
January becomes January plus February, and the snapshot ID changes. Schema
evolution is a capability to recognize, not a separate lab task in this
chapter.

## Concurrent Writers And Commits

Object storage is excellent for storing large immutable objects, but it is not a
traditional transactional database. An open table format adds transactional
behavior at the table-metadata layer.

A writer prepares new data files and metadata, then attempts to commit a new
table state. If another writer changed the table first, the commit can be
retried or rejected instead of silently corrupting the table. That is the
practical meaning of ACID semantics in this context: readers should see a
consistent table state, and failed writes should not become half-visible table
versions.

For this course, you only run small jobs. The principle still matters. Shared
data platforms need clear commit behavior because multiple jobs, users, and
tools may touch the same table.

## Alternatives

Apache Iceberg is not the only open table format.

Delta Lake is strongly associated with Databricks and Microsoft Fabric-style
lakehouse workflows. It is widely used and has a strong ecosystem.

Apache Hudi is often used for ingestion-heavy and update-heavy data lake
workloads, especially where incremental processing is central.

Iceberg is a good teaching choice here because it has broad engine support, a
clear metadata model, and a REST catalog pattern that fits our Kubernetes
platform.

The important lesson is not that every team must choose Iceberg. The important
lesson is that modern data lakes often need a table layer above the files.

## Machine Learning Connection

For AI work, the important word is version.

When you train a model, you should be able to record the dataset state that
produced it. "We trained on the taxi trip table" is weaker than "we trained on
snapshot 123456789 of
`course.preparation_for_ai_01.yellow_taxi_trips`".

That snapshot ID becomes model evidence. It helps with reproducibility,
debugging, audits, and comparisons between model runs.

This is also where data preparation starts to touch MLOps. You are not only
creating a dataset. You are creating evidence that connects a model artifact
back to a specific version of the data.

## Lab 4 Preview

In the lab, you use NYC Yellow Taxi trip records. The instructor prepares
cleaned random samples for January and February from the public monthly Parquet
files and stores those samples in the shared bucket. You turn the January
sample into an Iceberg table, query it, then append the February sample with
Spark.

The lab connects three ways of working with the same Iceberg table:

- PyIceberg gives you a Python entry point into the Iceberg catalog.
- DuckDB gives you a local SQL query engine.
- Spark gives you a shared compute engine for larger jobs.

The goal is not to memorize every Iceberg command. The goal is to feel the
difference between a path of Parquet files and a named table with history.

## Code Tools In The Lab

PyIceberg is the Python library you use to talk to Iceberg directly. It connects
to the Polaris REST catalog and uses PyArrow-based file IO to work with table
data and metadata.

DuckDB is the local SQL engine. Its `httpfs` and `iceberg` extensions let it
connect to S3-compatible object storage and the Polaris REST catalog.

Python is also used as the lab wrapper. The scripts load `participant.env`,
configure clients, generate Kubernetes manifests where needed, and write small
local plots so you can inspect what happened.

Spark is the shared-compute engine. The Spark job uses the Iceberg Spark
runtime, reads and writes through the Polaris catalog, and appends the February
sample to the same table.

Kubernetes runs Spark through `SparkApplication` objects. You submit jobs from
your participant VM using `participant.kubeconfig`.
