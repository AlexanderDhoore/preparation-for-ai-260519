# Lab 5: Multimedia Assets For Machine Learning

Start with [theory.md](theory.md), then review [slides.pdf](slides.pdf), then continue with the lab below.

In this lab you train neural networks from manifest-backed datasets in object
storage.

The shared bucket contains full manifests and assets for two real public
datasets:

- Food-101 images:
  `datasets/chapter05/food101/manifest.csv`
- Banking77 customer-service texts:
  `datasets/chapter05/banking77/manifest.csv`

You will not train on all classes at once. That would make the workshop slower
and less personal. Instead, you first explore the full dataset, then choose five
labels that you find interesting. The training script downloads and caches only
the assets for your chosen labels.

The practical challenge in this chapter is to keep looking at examples.
Neural-network training can feel abstract, but the artifacts are concrete:
image previews, text examples, training curves, confusion matrices, and
prediction examples.

## Start In The Course Environment

Run:

```bash
source .venv/bin/activate
```

Check that PyTorch can see the GPU:

```bash
python3 - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

If CUDA is available, the training scripts will use it. If not, they still run
on CPU, but image training will be slower.

## Lab 1: Food Image Classification

Food-101 contains photos of 101 food classes. Your job is to choose five of
those classes and fine-tune a pretrained image model on that smaller task.

### Step 1: Explore Food-101

Open:

```text
05-multimedia-assets/lab1/explore-food.py
```

Run:

```bash
python3 05-multimedia-assets/lab1/explore-food.py
```

First open:

```text
05-multimedia-assets/lab1/explore1_manifest_preview.csv
```

This is a small preview of the manifest. Each row points to one image object in
S3 and records the label, split, image size, and checksum. The manifest is the
dataset definition: without it, the bucket is only a large collection of image
files.

Look closely at the object paths too. They are partitioned with pieces such as
`split=train/label=apple_pie`. That does not replace the manifest, but it makes
the storage layout readable and lets tools copy or scan one split or label
without listing the entire dataset.

Then open:

```text
05-multimedia-assets/lab1/explore2_dataset_overview.png
```

This plot shows the size of the shared manifest. Food-101 is not a tiny toy
folder: it has many classes and many images. The class-balance panel shows
that each food class has the same number of examples. The image-size and
orientation panels come only from manifest metadata, but they already tell you
that the training code will need to resize images before batching them.

Then open:

```text
05-multimedia-assets/lab1/explore3_label_examples.png
```

This is your label menu: one example image for each Food-101 class. Pick five
labels that make a classification problem you would actually enjoy testing.
You can choose similar foods, such as desserts, or deliberately different foods,
such as `pizza`, `ramen`, and `greek_salad`.

The image grid also reminds you that images are not clean table rows. Lighting,
camera angle, plate style, background, and cropping all become part of what the
model sees.

### Step 2: Train A Food Classifier

Open:

```text
05-multimedia-assets/lab1/food-image-classification.py
```

You will find two TODOs:

- choose exactly five labels from `explore3_label_examples.png`
- fix one line in the PyTorch training loop

The model is MobileNet V3 Small from TorchVision. It is pretrained, which means
it has already learned general visual features from a large image dataset. The
script freezes the feature extractor and replaces the final classifier layer so
it predicts your five selected food labels.

The PyTorch training loop is the core pattern:

```text
image batch -> model -> predictions -> loss -> backward pass -> optimizer step
```

The TODO in the loop asks you to send the image batch through the model. That
single line is where tensors become model predictions.

Run the script once before fixing it:

```bash
python3 05-multimedia-assets/lab1/food-image-classification.py
```

It should fail with a TODO error or an unfinished model-call line. Fix the
TODOs, then run it again.

At the start, the script downloads and caches the selected images from object
storage. That is usually the slowest part of the run, so it prints progress
while the cache is filling. Once training starts, it prints the training loss
and accuracy, plus the validation loss and accuracy, after every epoch.

This lab pre-downloads the images because your selected Food-101 subset is small
enough to fit comfortably on local disk. For much larger image datasets, a
PyTorch dataloader can fetch images from object storage while training is
running. That is slower, especially if it repeats every epoch, but it lets you
train on datasets that are too large to fully copy into the local VM first.

First open:

```text
05-multimedia-assets/lab1/food1_selected_class_distribution.png
```

This plot shows how many training and validation examples each selected class
has. If you accidentally chose fewer or more than five classes, this plot makes
that obvious.

Then open:

```text
05-multimedia-assets/lab1/food2_training_curve.png
```

The four panels show training loss, validation loss, training accuracy, and
validation accuracy. Loss is the signal the optimizer minimizes. Accuracy is
the easier-to-read fraction of images classified correctly. You want training
loss to go down, but the validation panels matter most: they show whether the
model is also improving on images it did not train on.

Then open:

```text
05-multimedia-assets/lab1/food3_confusion_matrix.png
```

Rows are true labels. Columns are predicted labels. A strong model has most
counts on the diagonal. Off-diagonal counts show which foods the model confuses.

Finally open:

```text
05-multimedia-assets/lab1/food4_predictions.png
```

This shows eight validation examples for each selected class. Green prediction
labels are correct. Red prediction labels are mistakes. Look at the wrong
examples if you have any: are they genuinely confusing images, or does the
model make mistakes that seem surprising?

Questions:

- Did your five labels create an easy or hard classification problem?
- Which image classes are visually closest?
- Does the confusion matrix match what you see in the prediction examples?

## Lab 2: Banking Intent Classification

Banking77 contains short customer-service messages. The label is the customer's
intent: what the customer is trying to ask or solve.

This lab keeps the same manifest-to-training pattern as the image lab, but swaps
the image model for a transformer. A tokenizer splits text into tokens and turns
them into integer IDs. The transformer turns those IDs into contextual
representations, and the classifier head predicts the intent label.

### Step 1: Explore Banking77

Open:

```text
05-multimedia-assets/lab2/explore-banking.py
```

Run:

```bash
python3 05-multimedia-assets/lab2/explore-banking.py
```

First open:

```text
05-multimedia-assets/lab2/explore1_manifest_preview.csv
```

This is the same manifest idea as Lab 1. Each row points to one S3 object and
records the intent label, split, character count, checksum, and a short preview.
The preview is only there to make inspection easier. The full text still lives
as a separate object.

For very short text datasets like Banking77, you could store the full text
directly in a table and be perfectly practical. We still use the
manifest-plus-assets pattern here so the workflow stays consistent with images,
audio, PDFs, and larger text/document datasets where storing the full asset in a
CSV would become awkward.

The object paths are partitioned by split and label, just like the image paths.

Then open:

```text
05-multimedia-assets/lab2/explore2_dataset_overview.png
```

This plot has six panels. The first shows how many texts are in each split. The
second shows all intent labels sorted by number of examples, so you can see
whether the dataset is balanced. The third shows roughly how many words each
text message has. The fourth shows the train/validation split for every intent
label. The fifth shows which intents tend to have the longest messages. The
sixth checks whether train and validation texts have similar word-count
distributions. Use these panels later when you choose `MAX_TOKENS`.

Then open:

```text
05-multimedia-assets/lab2/explore3_label_menu.md
```

This is your label menu. It lists every intent label, the number of training
and validation examples, and one short example text. Pick five labels that are
related enough to be interesting but not identical. Several card and transfer
intents sound similar, which can make the confusion matrix more meaningful.

Read some example texts before training. Some labels are obvious from one
phrase. Others require more context. This is exactly why text models use
pretrained language representations instead of only counting words.

### Step 2: Train A Transformer Intent Classifier

Open:

```text
05-multimedia-assets/lab2/banking-transformer.py
```

You will find two TODOs:

- choose exactly five intent labels from `explore3_label_menu.md`
- choose a value for `MAX_TOKENS`

`MAX_TOKENS` controls how much text the tokenizer keeps for each example. A
token is usually a word, part of a word, or punctuation. It is not the same as a
letter. Token count is usually close to the word count, and often a little
higher because uncommon words can be split into subword pieces.

Use `explore2_dataset_overview.png` to estimate a reasonable value, then try a
few choices and compare the results. If the value is too small, useful words at
the end of a message can be cut off. If it is much larger than needed, training
does extra work on padding.

The transformer training loop has the same shape as the image loop, but the
batch contains token IDs and attention masks instead of image tensors:

```text
text -> tokenizer -> token IDs + attention mask -> transformer -> logits -> loss
```

Inside the loop, the script calls the transformer with tokenized inputs and
labels. That is the line where the pretrained language model becomes a
classifier for your selected intents. You do not have to edit that line, but it
is worth recognizing it:

```python
output = model(**inputs, labels=labels)
```

Run the script once before fixing it:

```bash
python3 05-multimedia-assets/lab2/banking-transformer.py
```

Fix the TODOs, then run it again.

The script caches the selected texts, then loads the Hugging Face tokenizer and
transformer model. The model download may be the slowest part the first time.
Once training starts, it prints loss and accuracy for both training and
validation after every epoch.

First open:

```text
05-multimedia-assets/lab2/banking1_selected_class_distribution.png
```

This plot confirms your five selected labels and their train/validation counts.

Then open:

```text
05-multimedia-assets/lab2/banking2_training_curve.png
```

Read this the same way as in the image lab: loss is the optimizer signal,
accuracy is the easier-to-read fraction of correct examples, and validation is
the split the model did not train on.

Do not be surprised if the validation accuracy becomes extremely high, even
perfect, for some label choices. Banking77 is a small, clean intent dataset, and
DistilBERT already understands a lot of English before this lab starts.

Then open:

```text
05-multimedia-assets/lab2/banking3_confusion_matrix.png
```

Use the confusion matrix to find intent labels that sound close to each other.
If the model is perfect, that is also evidence: your selected intents were easy
for this pretrained model to separate.

Then open:

```text
05-multimedia-assets/lab2/banking4_predictions.md
```

This file shows eight validation examples for each selected intent. Each example
is marked `correct` or `wrong`. If there are wrong predictions, check whether
the confused labels also sound close to you.

Finally open:

```text
05-multimedia-assets/lab2/banking5_embedding_map.png
```

This plot shows something different from the classifier score. A transformer
turns each text into a vector called an embedding. Texts with similar meaning
often get embeddings that are closer together. The script projects those
high-dimensional vectors down to 2D so you can inspect the shape visually.

Do not expect a perfect map. A 2D projection throws away information, and some
banking intents really are close in meaning. The useful question is whether
examples from related labels form nearby clouds or confusing overlaps.

Questions:

- Which intent labels are semantically close?
- Did your label choice create an easy or hard text problem?
- How is this training loop similar to the image lab?
- What changed because the assets are text instead of images?
- Does the embedding map agree with the mistakes in the confusion matrix?

## What You Learned

Both labs use the same platform pattern:

```text
shared manifest
  -> selected labels
  -> S3 assets
  -> local cache
  -> decoder/tokenizer
  -> tensors
  -> neural network
  -> features or embeddings
  -> classifier output
  -> visual or readable evidence
```

The asset type changes the decoder and the model architecture. The data-platform
pattern stays the same: object storage holds the bytes, and the manifest defines
the dataset.
