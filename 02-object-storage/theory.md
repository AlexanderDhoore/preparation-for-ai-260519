# Chapter 2: Object Storage And Data Lake Layers

Modern data lakes usually start with object storage. In this course, that is not
an optional detail. It is the backbone of the architecture.

The important idea is simple: store data as objects in a bucket, then build
processing and query systems on top.

Object storage is popular because it is:

- cost-effective for large raw data
- scalable for files, tables, and binary assets
- durable when configured correctly
- accessible from many tools through S3-style APIs
- close to the cloud platforms used in industry

After the first raw-data inspection, the mental model should shift away from
"my local folder" and toward "our shared data lake". Local files are useful for
the workshop smoke test, but object storage is where professional teams put raw
inputs, prepared layers, assets, metrics, and model artifacts.

Chapter 1 used local files on purpose. Chapter 2 moves into shared storage and
keeps using Python. You first learn how a data scientist reads, writes, lists,
and uses data that no longer lives on one laptop. Then you add a first
organization pattern on top: Bronze, Silver, and Gold layers.

Object storage also keeps large data out of Git. Git is excellent for source
code, small configuration files, and text that changes line by line. It is not
a good place for growing image datasets, audio files, video files, model
checkpoints, experiment logs, or large Parquet files. In a professional
machine-learning project, Git usually stores the code and configuration while
object storage stores the data and artifacts that the code produces.

## Why Not Put Everything In A Database?

Traditional SQL databases are excellent when you need transactions, indexes,
and many small updates.

Some raw data is tabular and can be imported into a SQL database. A CSV export
is the obvious example. But AI projects often collect more than clean tables:

- logs
- images
- audio
- video
- PDFs
- vendor files

Object storage is a good landing zone for that mixed raw data because it stores
files and binary assets directly. You can preserve the original inputs first,
then build query engines, catalogs, workflow tools, and Business Intelligence systems on top.

## Object Storage Is Not A Normal Filesystem

Object storage is different from the storage you use on a laptop.

With a local filesystem or network share, you think in directories and files.
You can open a file, modify a few bytes, and save it again. That model is
comfortable, but it becomes difficult to scale across many machines, many
teams, and very large datasets.

With block storage, you get something even lower-level: a virtual disk that an
operating system can format and mount. That is useful for databases and virtual
machines, but it is not the interface you want every data-processing script to
manage directly.

Object storage is higher-level. You store and retrieve whole objects through
an API. In practice, you should think of an object as a blob of bytes plus a
name and metadata. If you need a changed version of an object, you usually
write a new object rather than editing the old one in place.

That write-new-object habit is important for AI data. Raw data, model
artifacts, metrics, and validation reports often become evidence. You do not
want those files silently changing under the same name without a deliberate
versioning strategy.

## The S3 API Is The Contract

In industry, "S3" often means more than Amazon S3 itself. It usually means the
S3-style API: buckets, object keys, credentials, and operations such as upload,
download, list, delete, and inspect metadata.

That API became a de facto standard. Amazon S3 is the best-known service, but
many other systems expose a compatible or similar interface:

- Ceph RGW and MinIO for self-hosted object storage
- Google Cloud Storage and Azure Blob Storage or ADLS Gen2 in cloud platforms
- Cloudflare R2, Backblaze B2, Wasabi, and other storage providers

This is why the lab code uses an endpoint URL. If the code is written against
the S3-style API, the endpoint can point to AWS, Ceph, MinIO, or another
compatible service. The details differ, but the mental model and many client
libraries transfer.

At the API level, the basic operations are intentionally simple:

- `PUT` or upload writes an object
- `GET` or download reads an object
- `LIST` finds objects under a prefix
- `HEAD` reads object metadata without downloading the full object
- `DELETE` removes an object when policy allows it

For data work, `HEAD` is easy to underestimate. It lets code check whether an
object exists, how large it is, when it was last modified, and sometimes which
content type or custom metadata it carries without paying to download the full
file.

## Buckets, Objects, And Prefixes

Object storage does not really have folders. It has buckets and object keys.

A bucket is the top-level container. In this workshop, your participant bucket
is your writable workspace. The shared bucket is where prepared input datasets
live.

An object is one stored item: an image, an audio file, a JSON report, a Parquet
file, a model artifact, or any other blob of bytes. Every object has a key.

Example:

```text
datasets/engine-sounds/source=public/version=2026-05-01/manifest.csv
```

The slashes are part of the key. Tools display them like folders because that is
useful for humans.

This distinction matters when designing a data lake. A prefix is a naming
convention, not a directory with strong behavior. Object storage will not
automatically understand that `datasets/`, `assets/`, `reports/`, and
`models/` are different kinds of content. The team gives those names meaning
through code, access rules, documentation, and habits.

A useful key usually answers a few questions:

```text
what kind of object is this?
which project or use case owns it?
which source produced it?
which date, run, or version does it belong to?
```

That is why object naming is a design activity. The storage system is simple,
so the conventions around it need to be clear.

## Python Interface In This Chapter

Teams use object storage through many interfaces: command-line tools, cloud
platform UIs, SQL query engines, Spark jobs, and application code. In this
chapter, you start with Python because it is a direct fit for data-science
workflows.

Common patterns:

- use an S3 client to upload, list, and download objects
- use familiar Python tools such as `pandas` to load data from storage
- use PyTorch data loaders or custom readers to stream assets
- use Spark to read and write larger datasets later in the course

Some libraries read directly from S3-style URLs. Other libraries still expect a
local file, so the code downloads the object first or uses a filesystem adapter.
Both patterns are normal. In either case, the code should make the bucket and
object key visible, so it is clear which input was read and where the output was
written.

At the lowest level, Python code often looks like this:

```python
import boto3

s3 = boto3.client("s3", endpoint_url="https://s3.example.com")

s3.upload_file(
    Filename="02-object-storage/lab1/train2_confusion_matrix.png",
    Bucket="preparation-for-ai-01",
    Key="outputs/chapter02/lab1/train2_confusion_matrix.png",
)

response = s3.list_objects_v2(
    Bucket="preparation-for-ai-shared",
    Prefix="bronze/esc50/",
)
for item in response.get("Contents", []):
    print(item["Key"])
```

Higher-level libraries can hide some of that ceremony, but the same concepts
remain: bucket, key, credentials, permissions, and network access.

## Bronze, Silver, And Gold

Once object storage becomes the shared data lake, you need names for the
different states of your data. Bronze, Silver, and Gold is a common pattern for
that.

This is sometimes called a medallion architecture. It is not magic. It is a
naming convention that makes responsibilities clear:

```text
Bronze: raw data, received and traceable
  -> Silver: cleaned and standardized data
  -> Gold: data prepared for a specific use case
```

The pattern is useful because a data lake is more flexible than a traditional
data warehouse. A warehouse usually expects structured data early: the schema is
defined before loading, and the system is optimized for trusted business
questions. A data lake can accept raw files, semi-structured exports, images,
logs, audio, vendor formats, and tables. That flexibility is valuable for AI
projects, but it needs discipline.

Bronze/Silver/Gold gives that discipline a vocabulary.

A simple layout might look like this:

```text
bronze/project/source/
silver/project/domain/
gold/project/use-case/
```

The exact names matter less than the discipline:

- raw or received data has its own place
- cleaned data has its own place
- use-case-ready data has its own place
- prefixes make it clear which stage of preparation you are looking at

## Bronze

Bronze is the raw landing layer in the data lake.

Typical properties:

- original columns or fields are preserved
- minimal parsing is applied
- source filenames and ingestion timestamps are recorded
- bad records are not silently fixed
- checksums or source metadata may be stored next to the data

Bronze answers: what did we receive?

In this course, Bronze should be thought of as data written back to object
storage. It is not only a local folder and not only the file on someone's
laptop. It is the first shared version of what was received, stored in a place
where other tools can inspect it.

Bronze can contain many data types:

- CSV exports from machines
- JSON logs from software systems
- image folders
- audio snippets
- sensor files
- vendor-specific binary files

The important point is traceability. Bronze should make it possible to go back
to what was received before a cleaning rule changed it.

## Silver

Silver data is cleaned and standardized.

Typical properties:

- stable column names
- correct types
- deduplicated keys
- impossible values removed or flagged
- joins between related sources
- validated asset paths and generated metadata

Silver answers: what do we trust enough to analyze?

For tabular data, Silver might mean typed columns, normalized units,
deduplicated records, and consistent identifiers. For multimedia data, Silver
might mean validated file paths, checked checksums, extracted dimensions, audio
durations, transcripts, or a manifest table that lists all usable assets.

The key habit is that cleaning rules should be visible. A rule hidden in a
notebook cell is difficult to review. A rule in a named transformation script,
SQL model, or orchestrated task can be tested, discussed, and rerun.

## Gold

Gold data is prepared for a specific business or ML purpose.

Typical properties:

- feature tables
- dashboard tables
- training-ready datasets
- clear labels and splits
- metrics for validation

Gold answers: what can we use for decisions or models?

Gold is where the AI connection becomes explicit. A Gold table might be a
training dataset, a feature table, a dashboard table, or an evaluation dataset.
Gold does not always need to be one table. For a computer vision project, Gold
might be a manifest table plus image files in object storage. For a forecasting
project, Gold might be a time-series feature table. For a Business Intelligence dashboard, Gold
might be an aggregated business table.

That means "Gold" is not the same as "final forever". It means "fit for this
use case". A company can have several Gold datasets derived from the same
Silver layer:

```text
gold/quality-dashboard/
gold/failure-prediction-training/
gold/monthly-management-report/
gold/computer-vision-manifest/
```

## Training From Layered Object Storage

Once files are in object storage, a model training script should not depend on
a private laptop folder. It should read from a shared location and write its
outputs back to a shared location.

For machine learning, the bucket can hold:

- Bronze raw data that should not be overwritten
- Silver cleaned data and metadata
- Gold training-ready datasets
- images, audio, video, scans, and other assets
- validation reports
- model metrics
- trained model artifacts

In most projects, training should read from Gold, not directly from raw input.
The Gold layer is where the team has made deliberate choices about features,
labels, splits, asset manifests, and validation checks.

The first version can still be simple:

```text
read Gold data from object storage
  -> train a small model
  -> write metrics and artifacts back to object storage
```

Layering also supports reproducibility. If a model changes, the team should be
able to find out whether the cause was:

- a different raw source
- a changed cleaning rule
- a changed label
- a changed feature
- a changed model

The model training step should never be a mystery. A team should be able to
say:

```text
this model used Gold dataset version X
Gold version X came from Silver version Y
Silver version Y was prepared from Bronze source Z
```

This is the bridge from basic scripting to professional data preparation. The
same reasoning applies to dashboards. If a Business Intelligence number changes, the team should
be able to ask whether the change came from the source, the cleaning rules, the
aggregation logic, or the dashboard filter.

## Audio In Object Storage

The lab of Chapter 2 focuses on audio objects. These are deliberately heavier
than the first chapter's local tabular examples. Python usually reads a manifest
table first, then loads the binary assets referenced by that manifest.

That pattern scales better than passing folders around:

- object storage holds the files
- a table or manifest describes what the files mean
- the training code reads both
- outputs go back to a stable prefix

For assets, the training code usually reads a manifest first:

```text
asset_path,label,split
s3://preparation-for-ai-shared/bronze/esc50/audio/dog/0001.wav,dog,train
s3://preparation-for-ai-shared/bronze/esc50/audio/rain/0001.wav,rain,train
```

That separation is important. The object storage key tells the code where the
bytes are. The manifest tells the code what the bytes mean.

## Storage And Compute Stay Separate

Bronze/Silver/Gold is not tied to one compute tool.

Object storage keeps the layers. Python, DuckDB, Spark, Airflow, Business Intelligence tools, and
MLflow-style tracking can read and write those layers. The storage remains
stable even when the compute tool changes.

That is why this pattern appears again later:

- Efficient tabular data writes Gold Parquet feature tables.
- Open table formats add snapshots and schema evolution to those Gold tables.
- Multimedia assets use manifests to connect Gold datasets to large objects.
- Orchestration automates the steps between layers.
- Validation and Business Intelligence make layer quality visible.
- MLOps records which prepared data version trained which model.

## Permissions Matter

Object storage is simple, but it still needs security.

Typical production rules:

- many people can read curated data
- fewer systems can write Silver or Gold data
- raw data is protected from accidental overwrite
- service accounts are scoped to the prefixes they need
- sensitive datasets are encrypted and audited

In the workshop, the setup is simpler. The professional habit is the same:
think about who can read, who can write, and what should never be overwritten.

## Cost And Lifecycle

Object storage is cheap, but it is not free.

The main cost drivers are usually:

- stored bytes
- requests
- data transfer
- keeping too many old copies forever
- repeatedly reading large assets during training

That is why teams use lifecycle rules. A lifecycle rule can move older data to
cheaper storage, expire temporary outputs, or keep raw data longer than derived
data.

The policy depends on the project. Raw source data may need to be kept because
it is evidence. Temporary experiment outputs can often be deleted sooner.
Curated Gold datasets and model artifacts may need clear retention because they
explain important results.

Good object-storage design is therefore not only a folder layout. It is also:

- access rules
- overwrite rules
- retention rules
- lifecycle rules
- cost awareness

## Versioning Needs A Clear Choice

Object storage can support versioning, but you need to know what kind of
versioning you are using.

The simplest approach is naming convention versioning:

```text
bronze/esc50/version=2026-05-19/manifest.csv
models/engine-condition/run=2026-05-19T101500/model.onnx
outputs/chapter02/run=alexander-01/train2_confusion_matrix.png
```

This is easy to understand and works well in many projects. The downside is
that the discipline lives in your code and team habits. If someone overwrites a
key accidentally, the naming convention alone cannot save you.

Many S3-compatible systems also support native object versioning. That can
protect older versions of a single object key. It is useful as a safety net,
but it is still object-level versioning. It does not automatically give you a
clean snapshot of a full dataset with hundreds or thousands of objects.

For dataset-level versioning, teams often add another layer: manifests, open
table formats such as Iceberg, DVC, or LakeFS. The common theme is that a
dataset version is more than one file. It is a documented set of object keys,
schemas, checksums, labels, splits, and metadata.

## How This Maps To Cloud Platforms

The hands-on course uses S3-compatible storage on VIVES infrastructure. That
keeps the workshop open source and self-hosted, but the architecture is not a
VIVES-only idea. The same mental model appears in the large cloud platforms.

On AWS, the closest storage service is Amazon S3. You still think in buckets,
objects, keys, prefixes, IAM permissions, lifecycle policies, and object
versioning. Other AWS services can then read from or write to S3: analytics
engines, Spark clusters, serverless jobs, Business Intelligence tools, and machine-learning
services.

On Azure, the storage layer is usually Azure Blob Storage or Azure Data Lake
Storage Gen2. ADLS Gen2 is built on Blob Storage and adds a hierarchical
namespace that makes data-lake workloads feel more filesystem-like while still
using cloud object storage as the foundation. The surrounding platform might be
Azure Databricks, Microsoft Fabric, Synapse, Data Factory, or Azure Machine
Learning.

On Google Cloud, the comparable service is Cloud Storage. You again work with
buckets and objects. Slashes in object names are often displayed as folders by
tools, but the important storage concept is still the object key. Query engines,
Dataproc, Dataflow, BigQuery, and Vertex AI can use Cloud Storage as part of a
larger data and ML platform.

In this workshop, the names are different:

- Ceph RGW provides the S3-compatible object storage.
- Python, Polars, DuckDB, Spark, and Airflow provide the compute side.
- Polaris and Iceberg provide table metadata later in the course.
- Superset and MLflow make data quality and model evidence visible.

You do not need to memorize every cloud product name now. The transferable
pattern is the important part: shared object storage, explicit permissions,
clear prefixes, lifecycle rules, processing engines on top, and reproducible
outputs written back to stable locations.

## Lab 2: Object Storage And Data Lake Layers

The Chapter 2 lab keeps the code practical. You are still using Python scripts,
but the files no longer live only in your local folder.

The workshop platform uses two kinds of buckets:

- one personal bucket per participant, such as `preparation-for-ai-01`
- one shared read-only bucket, `preparation-for-ai-shared`

That split is deliberate. Shared datasets should be stable. Participant outputs
should be isolated. If everyone writes into the same place, a workshop becomes
a miniature version of a badly governed data lake.

You will use `boto3` as the low-level S3 client. This is the library that lists
objects, downloads bytes, uploads plots, and makes the bucket/key model
visible in code:

```python
response = client.get_object(Bucket=shared_bucket, Key=object_key)
content = response["Body"].read()
```

You will use `python-dotenv` to load workshop credentials from
`participant.env`. This keeps access keys out of the scripts and makes it
clear which bucket is shared and which bucket is your own writable workspace.

You will use `pandas` again, but now the dataframe often starts as a manifest
read from object storage. The manifest connects object keys to labels,
splits, categories, and other metadata.

For audio assets, you will inspect WAV metadata from bytes, choose a subset of
classes for your own Silver layer, extract simple audio features, and train a
small model from a Gold feature table. This makes a useful distinction visible:
Bronze can contain the original WAV files, Silver can contain a curated subset
for one task, and Gold can contain the numeric representation that a model
actually consumes.

The important pattern is:

```text
read Bronze objects and manifests from the shared bucket
  -> create a local artifact you can inspect
  -> write Silver audio and Gold features to your participant bucket
  -> upload plots to your participant bucket
```

That is already the shape of a data platform. The objects are shared, the
meaning comes from metadata, and your outputs go back to stable locations.
