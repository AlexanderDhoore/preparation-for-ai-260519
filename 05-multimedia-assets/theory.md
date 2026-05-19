# Chapter 5: Multimedia Assets For Machine Learning

AI datasets are not always neat tables.

The first part of this course focused on tabular data because tables are the
easiest place to teach the data-lake pattern. Object storage gives us a shared
landing zone. Parquet makes tabular data efficient. Query engines let us work
with files as if they were database tables. Open table formats add snapshots,
schema evolution, and transactional table state.

Multimedia data changes the shape of the problem.

An image, audio file, video, PDF, point cloud, or hyperspectral scan is usually
not something you want to put directly inside a SQL table. The file can be
large. The format can be specialized. The training code may need to decode it
batch by batch. The useful metadata around the file is often much smaller than
the file itself.

A common beginner pattern is to organize assets by folders:

```text
train/healthy/image001.jpg
train/rust/image002.jpg
test/healthy/image003.jpg
```

That is convenient for a first experiment. Many libraries can read this layout.
It is also fragile.

The folder name becomes the label. The parent folder becomes the split. If
someone moves a file, renames a folder, or copies a subset, the dataset meaning
changes. The file itself does not know that happened.

Professional data preparation makes the meaning explicit:

```text
asset_uri,label,split,sha256
s3://preparation-for-ai-shared/datasets/chapter05/food101/assets/split=train/label=pizza/0001.jpg,pizza,train,...
s3://preparation-for-ai-shared/datasets/chapter05/food101/assets/split=train/label=ramen/0002.jpg,ramen,train,...
```

That table is called a manifest. The object key tells code where to find the
bytes. The manifest tells code what those bytes mean.

That is the central pattern of this chapter:

```text
large binary assets live in object storage
small structured meaning lives in a manifest table
training code reads both
```

For machine learning, the manifest is part of the dataset definition.

## Assets Are Not Rows

Multimedia datasets often start as files rather than rows:

- photographs from a camera
- microscope or inspection images
- audio recordings
- videos or frame sequences
- scanned documents and PDFs
- text documents and tickets
- point clouds
- hyperspectral cubes
- vendor-specific binary exports

Those files may eventually produce tables. You might extract image features,
audio durations, transcript embeddings, bounding boxes, or quality metrics. But
the original file remains important. It is the source of the derived
representations.

That is why the asset layer and the metadata layer are usually separate:

- the asset layer: the files themselves
- the metadata layer: the structured description of those files

A file path can tell you where bytes are stored. It cannot reliably tell you
the label, split, source, consent status, checksum, preprocessing history, or
model-ready interpretation of those bytes.

## The Manifest Pattern

The manifest columns depend on the asset type.

For images, a manifest might contain:

- `asset_uri`: where the image lives in object storage
- `label`: the class label
- `split`: train, validation, or test
- `sha256`: the expected checksum
- `content_type`: the media type
- `width` and `height`: basic image properties
- `source_dataset`: where the record came from

For audio, useful fields include:

- `duration_seconds`
- `sample_rate_hz`
- `channels`
- `codec`
- `transcript`
- `speaker_id` or another grouping key

For text and documents, useful fields include:

- `language`
- `document_type`
- `source_id`
- `chunk_id`
- `contains_pii`
- `label`
- `split`

For scientific assets, the manifest can become even more important. A
hyperspectral image may consist of a header file, a binary file, and calibration
metadata. A point cloud may need coordinate-system information. A medical image
may need acquisition metadata and strict privacy controls.

The manifest is the bridge between object storage and model training:

```text
manifest row
  -> object-storage URI
  -> decoded asset
  -> label, target, or metadata
  -> model input
```

If the manifest is good, many tools can use the same dataset definition:

- Python scripts can load assets for inspection.
- Spark jobs can validate file existence and metadata.
- SQL engines can query labels, splits, and counts.
- Business Intelligence dashboards can show dataset composition.
- PyTorch dataloaders can feed training batches.
- MLflow or another tracking tool can record which manifest version trained a
  model.

You can still use readable folder layouts. Humans like them. They are useful
for browsing and debugging. The point is that the folder layout should not be
the only source of truth.

## Versioning Asset Datasets

Versioning one object is not the same as versioning a dataset.

S3-compatible object stores can often keep object versions. That helps with
recovery, but a model usually trains on many objects and a manifest. The team
needs to know which complete set of files, labels, splits, and preprocessing
decisions belonged together.

There are several workable strategies:

- immutable object keys for original assets
- new prefixes for new derived versions
- manifest files with checksums and split information
- Parquet manifests for efficient querying
- Iceberg tables for versioned manifests and snapshots
- DVC or lakeFS for Git-like data versioning workflows

For a small workshop lab, a checksummed manifest file is enough. For a larger
team, an Iceberg manifest table becomes attractive because it gives the
metadata table the same snapshot behavior you saw in Chapter 4.

The asset files stay in object storage. The versioned table describes which
assets are part of a dataset version.

That is the key idea:

```text
version the manifest
keep the heavy assets stable
record the relationship between them
```

## Training From A Manifest

In a model-training loop, you usually do not read all multimedia assets at once.
You read batches.

In PyTorch, the pattern is:

```text
manifest table
  -> Dataset object
  -> DataLoader
  -> model training loop
```

The `Dataset` knows how to turn one manifest row into one training example. It
loads the asset, decodes it, applies transforms, and returns tensors and labels.

The `DataLoader` batches examples, optionally shuffles them, and can use worker
processes to load data in parallel.

That means the data path matters:

- object storage throughput matters
- image or audio decoding speed matters
- batch size matters
- prefetching matters
- local caching can matter
- GPU utilization depends on the input pipeline

A GPU can only train while it has data. If the GPU waits for files to download,
decode, or transform, the expensive part of the machine sits idle.

That does not mean every dataset must be copied fully to local disk. It means
the data pipeline should be intentional. For small datasets, direct reads from
object storage are fine. For larger datasets, you may cache files locally,
preprocess assets into training-friendly formats, or use a streaming dataset
format.

## File Formats And Compression

For multimedia data, file format is part of the data engineering design.

The exact format depends on the modality. Images may use JPEG, PNG, TIFF, WebP,
or camera-specific raw formats. Audio may use WAV, FLAC, MP3, Opus, or
telephony encodings. Video uses containers and codecs. Scientific assets may
use formats such as HDF5, NetCDF, Zarr, or ENVI header/binary pairs.

You do not need to memorize all of those names. The important question is what
the format does to storage cost, network traffic, decoding speed, and signal
quality.

In object storage workflows, compression is usually a good idea. Compressed
files cost less to store and are faster to move over the network. Decompression
does take CPU time, but modern CPUs are often fast enough that the smaller
download more than compensates for the decompression step. In many data
platforms, network and storage I/O are the bottleneck before raw CPU
decompression is.

That does not mean every compression choice is equally good.

Lossless compression preserves the exact signal. For machine learning, this is
often the safer default. The model sees the original information, while the team
still saves storage and bandwidth. PNG, FLAC, and some scientific formats can
play this role, depending on the modality.

Lossy compression changes the signal. JPEG, MP3, many video codecs, and some
modern image formats can produce much smaller files, which can make upload,
download, and training input pipelines faster. The tradeoff is that the model
now sees compression artifacts. Sometimes that is acceptable. Sometimes it is a
serious problem.

For example, lossy compression might be fine for a rough visual classification
dataset where the target is robust to small artifacts. It may be a bad choice
for microscopy, medical imaging, hyperspectral data, small defect detection, or
any task where subtle signal changes matter.

So the practical advice is:

- do compress data when it reduces storage and I/O cost
- prefer lossless compression when the original signal matters
- use lossy compression only when the artifact tradeoff is understood
- record the format, codec, resolution, and preprocessing choices in the
  manifest

For AI, "smallest file" is not automatically the best answer. The best format
is one that preserves the information the model needs and can still be read
fast enough by the training pipeline.

## Checksums And Fingerprints

Object storage is durable, but durability does not tell you whether a dataset
is still the dataset you expected.

A manifest should often include a checksum or fingerprint. A checksum such as
SHA-256 answers a simple question:

```text
are these bytes exactly the bytes we recorded?
```

That matters when:

- files are re-uploaded
- files are converted or resized
- a dataset is copied between buckets
- training is repeated months later
- multiple people work on the same asset collection

Checksums are strict. If one byte changes, the checksum changes. That is useful
for reproducibility.

Sometimes you also need softer fingerprints. For images, you may store width,
height, mode, and maybe a perceptual hash. For audio, duration and sample rate
are useful quick checks. For text, you may store character count, language, or
document hash.

The goal is not to collect metadata for its own sake. The goal is to make
unexpected changes visible before they reach model training.

## Splits, Groups, And Leakage

The manifest is also where train, validation, and test splits should become
explicit.

For multimedia data, leakage is easy to miss. A model can look good because the
same real-world object appears in both train and test under slightly different
files.

Examples:

- several images of the same plant leaf
- multiple frames from the same video
- repeated recordings from the same speaker
- document chunks from the same original report
- multiple scans of the same food sample
- augmented copies of one image

If the split is only implied by folders, it is hard to audit. If the split is a
manifest column, you can query it.

You can also include grouping keys:

- `sample_id`
- `source_document_id`
- `video_id`
- `speaker_id`
- `device_id`
- `batch_id`

Those grouping keys help you make better split decisions. For example, all
frames from the same video should usually stay in the same split. All chunks
from the same document should usually stay together. All scans from one
physical sample may need to stay together.

Good splits are not a model detail. They are a data preparation decision.

## Original Assets, Derived Assets, And Caches

Multimedia pipelines often create derived files.

An original image might produce:

- a resized training image
- a thumbnail for dashboards
- a segmentation mask
- an embedding vector
- a quality-control report

An audio file might produce:

- a resampled WAV file
- a spectrogram image
- a transcript
- a voice-activity mask
- a feature vector

A document might produce:

- extracted plain text
- cleaned text
- chunks
- embeddings
- redacted versions

Do not overwrite the original asset with a derived asset unless you have a very
good reason. Keep the original stable. Write derived assets to their own
prefixes and record how they were created.

A useful layout is:

```text
datasets/chapter05/food101/assets/
datasets/chapter05/food101/manifest.csv
datasets/chapter05/banking77/assets/
datasets/chapter05/banking77/manifest.csv
```

The exact prefixes can differ. The principle is more important than the names:
asset bytes and metadata should not be mixed accidentally. If you later create
derived assets, such as resized images, thumbnails, transcripts, or embeddings,
write those derived objects to their own prefixes and record how they were
created.

Local caches are different again. A cache is allowed to disappear. It exists to
make training faster, not to define the dataset. If a cache is deleted, the
dataset should still be reproducible from object storage and the manifest.

## Alternative Versioning Systems

The manifest pattern is not the only way to version asset datasets.

Some teams use tools that add Git-like workflows around large files and object
storage. These tools can be very useful, especially when the team wants
branches, commits, diffs, and reproducible checkouts for data.

DVC works close to Git. The Git repository stores small pointer files and
metadata. The large data files live in a remote storage location, such as S3.
That makes it natural for teams that already think in terms of Git commits,
pull requests, and experiment folders.

lakeFS works closer to the object store itself. It adds repository, branch,
commit, and merge concepts on top of object storage. Instead of manually
choosing prefixes for every dataset version, the team can work with a data-lake
repository that behaves more like a version-controlled filesystem.

Git LFS and git-annex are related ideas from the software-development world.
They keep large binary files out of normal Git history and replace them with
references to externally stored content. They can be useful for smaller asset
collections, but they are usually less central in modern data-lake workflows
than DVC, lakeFS, or table-format metadata.

These systems solve real problems, but they also change the platform contract.
If a dataset version only exists as a DVC checkout or a lakeFS branch, every
tool in the workflow needs to understand that layer, or the team needs glue
code that translates it back to normal object paths and tables.

That is the main tradeoff.

Open manifests, Parquet tables, and Iceberg tables keep the dataset definition
visible to many engines: Python, SQL, Spark, dashboards, and training code. DVC
or lakeFS can give stronger version-control workflows, but the team becomes
more dependent on that versioning system.

For this course, we use the manifest-first approach because it is simple,
transparent, and easy to connect to the rest of the data platform. In a larger
project, DVC or lakeFS can still be the right choice if the version-control
workflow is worth the extra platform dependency.

## Lab 5 Preview

In the labs, you use two real public datasets that represent different asset
types:

- Food-101 for image classification
- Banking77 for text intent classification

Both labs start from the same pattern. The full course datasets live in the
shared object store. A manifest table defines the labels and splits. You first
inspect a manifest preview, then explore the full dataset shape, then choose a
smaller set of labels for your own training run. The training script filters
the manifest, reads only the selected assets, caches them locally, and trains a
neural network on the GPU VM.

Banking77 is intentionally a borderline case for this pattern. Its texts are
short enough that storing them directly in a table would also be reasonable.
The lab keeps the texts as separate objects so the same manifest idea carries
across images, audio, text files, PDFs, and larger document collections.

In the image lab, TorchVision loads a pretrained MobileNet model. The script
replaces the final classifier layer so it predicts the five food labels you
selected. The model has already learned general visual features, so the lab can
focus on how assets, transforms, batches, labels, and outputs connect.

In the text lab, Transformers loads a pretrained DistilBERT model. The tokenizer
turns text into token IDs, and the classifier predicts the five banking intents
you selected. This is the same manifest idea again, but the decoder and model
architecture are different. The lab also visualizes transformer embeddings:
high-dimensional vectors that represent text before the final classification
head turns them into label scores.

The point is not to build state-of-the-art models. The point is to see that a
modern neural-network training loop still needs a clear dataset definition:

```text
manifest -> asset bytes -> decoder -> tensors -> model -> evidence
```

Inside the model, pretrained layers turn tensors into useful features. In the
text lab, you inspect those features as transformer embeddings before the final
classifier head turns them into label scores.

## Code Tools In The Lab

You will mostly use Python in this chapter.

`boto3` gives Python access to S3-compatible object storage.

`pandas` and `pyarrow` are useful for reading and writing manifest tables.

`Pillow` opens images and extracts image metadata such as width and height.

`torch` provides tensors, GPU execution, neural-network layers, losses, and the
training loop.

`torchvision` provides pretrained image models and standard image transforms.

`transformers` provides pretrained language models and tokenizers.

The code should write evidence that is easy to inspect:

- class and split counts
- image previews or text prediction examples
- training curves
- confusion matrices

Those outputs become evidence that the dataset definition is understandable
before it is used for training.
