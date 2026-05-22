# Lab 2: Object Storage And Data Lake Layers

Start with [theory.md](theory.md), then review [slides.pdf](slides.pdf), then continue with the lab below.

Chapter 1 used local CSV files. In this lab, the source data already lives in
shared S3-compatible object storage.

You will work with one environmental audio dataset and build a small data lake
workflow around it:

```text
shared Bronze audio objects
  -> your Silver subset
  -> your Gold feature table
  -> a model trained from Gold
```

The important habit is to keep the location visible. When a script reads from
object storage, you should be able to point to the bucket and object key. When
a script writes an output, you should know whether it is local, in your bucket,
or both.

## Prepare Your Environment

Your VM already contains a `participant.env` file in the repository root. This
file contains the object-storage settings for your personal workshop account:

- the S3 endpoint
- the shared read-only bucket
- your own participant bucket
- the access key and secret key used by the lab scripts

Open the file once before you start:

```bash
cat participant.env
```

Do not paste these values into notebooks, reports, or screenshots. The point of
the environment file is that scripts can load credentials without hard-coding
secrets in Python code.

Then activate the course Python environment:

```bash
source .venv/bin/activate
```

## Dataset

This lab uses ESC-50, a public environmental sound dataset. The shared Bronze
layer contains WAV audio clips and a CSV manifest:

```text
s3://preparation-for-ai-shared/bronze/esc50/manifest.csv
s3://preparation-for-ai-shared/bronze/esc50/audio/...
```

The manifest is a table. Each row describes one audio object: its label, fold,
object key, content type, byte size, and checksum. The audio bytes live as
objects in S3. The manifest tells code what those objects mean.

## Step 1: Explore Bronze Audio

Before you run the first lab script, open the shared S3 helper:

```text
02-object-storage/s3_lab_common.py
```

Find the `load_s3()` function. The endpoint URL and bucket names are already
loaded near the top of the function. In the `boto3.client(...)` call, fix the
two TODO placeholders for the access key and secret key by using the exact
variable names from `participant.env`.

Open:

```text
02-object-storage/lab1/explore-audio.py
```

Run:

```bash
python3 02-object-storage/lab1/explore-audio.py
```

This script reads the Bronze manifest from the shared bucket, downloads a small
sample of WAV files, and creates local outputs you can inspect.

First open:

```text
02-object-storage/lab1/explore1_bronze_manifest.csv
```

This is a local copy of the Bronze manifest. It is the table that connects
labels to audio objects in the shared bucket. The audio itself is not inside
this CSV file; the `key` column tells the script where each WAV object lives in
object storage.

Questions:

- Which columns describe the object location?
- Which columns describe the label?
- What does the `fold` column probably help with later?

Then open:

```text
02-object-storage/lab1/explore2_class_counts.png
```

This plot shows the available ESC-50 classes and how many clips each class has.
Use it as the menu for the next step.

Questions:

- Which classes sound interesting to you?
- Can you find groups such as animals, alarms, weather, machines, or human sounds?
- Are the class counts roughly balanced?

Then open:

```text
02-object-storage/lab1/explore3_audio_metadata.png
```

This plot summarizes technical metadata from sampled audio objects: duration,
sample rate, and object size. This is the kind of check you do before trusting a
folder full of media files.

Questions:

- Do the sampled clips have a consistent duration?
- Do they have a consistent sample rate?
- Why would inconsistent audio metadata make later processing harder?

Then open:

```text
02-object-storage/lab1/explore4_waveform_samples.png
```

This plot shows waveform examples. A waveform is the audio signal over time. It
does not tell the whole story, but it gives you a first feeling for loudness,
silence, bursts, and repeated patterns.

Finally, listen to a few local sample clips:

```text
02-object-storage/lab1/explore_sample_dog.wav
02-object-storage/lab1/explore_sample_rain.wav
02-object-storage/lab1/explore_sample_chirping_birds.wav
```

The point of this step is simple: before you build Silver or Gold data, you
should know what the Bronze data actually contains.

## Step 2: Build A Silver Audio Subset

Open:

```text
02-object-storage/lab1/silver-audio-layer.py
```

You will find one TODO:

- choose exactly five ESC-50 labels from the Bronze dataset

Pick labels that make an interesting classification problem. You might choose
five animal sounds, five machine sounds, or a deliberately mixed set.

Run the script once before fixing it:

```bash
python3 02-object-storage/lab1/silver-audio-layer.py
```

It should fail with a `NotImplementedError`. Fix the TODO, then run it again.

When fixed, the script copies only your selected classes from the shared Bronze
bucket into your participant bucket and writes a new Silver manifest:

```text
silver/chapter02/esc50/audio/<label>/<clip>.wav
silver/chapter02/esc50/audio_manifest.csv
```

Open:

```text
02-object-storage/lab1/silver1_selected_class_counts.png
```

This plot confirms what moved into Silver.

Questions:

- Did you get exactly the five classes you selected?
- Are there enough clips per class to train a small model?
- What changed between Bronze and Silver?

Bronze is the received dataset. Silver is the curated subset for your current
task. You did not edit audio by hand; you created a cleaner dataset definition.

## Step 3: Build Gold Audio Features

Open:

```text
02-object-storage/lab1/gold-audio-features.py
```

The script extracts numeric features from every Silver audio clip. Examples:

- signal energy
- peak amplitude
- zero-crossing rate
- spectral centroid
- low, mid, and high frequency shares

Run:

```bash
python3 02-object-storage/lab1/gold-audio-features.py
```

The script writes a Gold feature table to your participant bucket:

```text
gold/chapter02/esc50/audio_features.csv
```

It also writes a local copy:

```text
02-object-storage/lab1/gold_audio_features.csv
```

First open:

```text
02-object-storage/lab1/gold1_feature_overview.png
```

This plot compares each extracted feature across your five selected labels.
Look for features where the boxes sit in different ranges for different labels.
Those features may help a classifier.

Then open:

```text
02-object-storage/lab1/gold2_feature_correlation.png
```

This plot shows which features move together. Highly correlated features may be
partly redundant. That is not automatically bad, but it is useful to notice
before choosing model inputs.

Gold no longer contains audio bytes. It contains a model-ready table derived
from audio.

## Step 4: Train From Gold Features

Open:

```text
02-object-storage/lab1/train-audio-model.py
```

You will find one TODO:

- choose feature columns for the model

Use the Gold plots from Step 3. Start with three to six features that look
useful. Then run the script again with a different feature set and compare the
results.

Run the script once before fixing it:

```bash
python3 02-object-storage/lab1/train-audio-model.py
```

It should fail with a `NotImplementedError`. Fix the TODO, then run it again.

First open:

```text
02-object-storage/lab1/train1_accuracy.png
```

This plot compares three results:

- a majority-class reference
- your chosen feature set
- an all-feature reference

The majority-class reference is the boring model that always predicts the most
common class. Your model should beat it. The all-feature reference shows what
happens when the same RandomForest classifier gets every extracted feature.

Then open:

```text
02-object-storage/lab1/train2_confusion_matrix.png
```

This file contains two confusion matrices: your chosen-feature model first, and
the all-feature reference below it. Rows are true labels. Columns are predicted
labels. This lets you see whether extra features help every class or only some
classes.

Questions:

- Which classes are easy?
- Which classes are confused?
- Which mistakes disappear in the all-feature reference?
- Does that match what you heard in the sample audio?

Finally open:

```text
02-object-storage/lab1/train3_feature_importance.png
```

This file also contains two panels. The first panel shows which of your selected
features the RandomForest used most often. The second panel shows the same idea
for the all-feature reference.

Questions:

- Did the model use the features you expected?
- Does the all-feature reference depend on features you did not choose?
- Did a frequency feature matter more than an amplitude feature?
- Would you change your feature list and rerun?

## What You Learned

You used object storage as a data lake, not just as a file dump:

```text
Bronze: shared raw audio objects and source manifest
Silver: your selected classes and curated manifest
Gold: numeric feature table for modelling
```

You also saw the manifest pattern in practice. The audio bytes stayed in object
storage. The CSV manifests and feature tables described which objects matter,
where they live, and how they can be used by later code.
