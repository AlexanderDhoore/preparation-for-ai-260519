# Lab 1: AI-Ready Data Starts Locally

Start with [theory.md](theory.md), then review [slides.pdf](slides.pdf), then continue with the lab below.

This chapter starts with local files and Python. That is intentional. Before a
team builds a data lake, it should know what kind of data it has, what the data
means, and what can go wrong before a model ever starts training.

In this lab you will do two small modelling exercises, but the models are not
the only point. You will inspect the raw CSV files, find the target columns,
choose input features from plots, create a new target for an engine time-series
dataset, and compare model results that look good for very different reasons.

The technical questions are:

- Which column is the thing the model should predict?
- Which input columns seem useful before you train anything?
- How much accuracy do two carefully chosen columns give compared with all
  available columns?
- How do you turn engine sensor logs into a remaining-life prediction target?
- What changes when a split leaks rows from the same engine into train and test
  data?

## Lab Overview

- **Lab 1:** explore a real forest cover type dataset, choose two useful
  columns, train a small random forest classifier, and compare it with a full
  model.
- **Lab 2:** inspect a real engine time-series dataset, create a derived
  target, and compare an honest split with a leaky random row split.

The datasets are downloaded from Hugging Face and cached inside this chapter:

```text
01-ai-ready-data/covertype.csv
01-ai-ready-data/turbofan.csv
```

The cache is intentionally not committed to Git.

## Start Python From The Course Environment

The course uses one Python virtual environment in the repository folder. A
virtual environment is a local Python installation with the packages for this
course, such as pandas, scikit-learn, and matplotlib. It keeps the workshop
packages separate from the system Python on the VM.

When you open the course folder in VS Code, the terminal should already start in
`/root/preparation-for-ai`. Activate the environment before running Python
commands:

```bash
source .venv/bin/activate
```

After activation, your prompt should start with `(.venv)`. That means commands
such as `python3 ...` will use the course packages.

If your terminal is somewhere else, first move back to the course folder:

```bash
cd /root/preparation-for-ai
```

## Prepare The Public Datasets

Run:

```bash
python3 01-ai-ready-data/download_datasets.py
```

This downloads:

- `mstz/covertype`
- `nominal-io/nasa-turbofan-degradation`

Both are real public datasets. They are small enough for a workshop, but they
already show professional problems: targets, many columns, class imbalance,
time-dependent rows, and the risk of accidentally evaluating the wrong thing.

Checkpoint:

```bash
ls -lh 01-ai-ready-data/covertype.csv 01-ai-ready-data/turbofan.csv
```

You should see one CSV file for forest cover type and one CSV file for NASA
turbofan data. Generated reports and figures are written inside the lab folders
later.

## Lab 1: Forest Cover Type Classification

Lab 1 has three steps. First you explore the dataset. Then you train a model
from only two columns. Finally, you train a big model and compare the results.

Before you run the first script, read the dataset guide:

```text
01-ai-ready-data/lab1/dataset-guide.md
```

Keep it open while you inspect the CSV and the plots. The column names are
short, and the guide tells you what the measurements and target classes mean.

### Step 1: Explore The Dataset

Open:

```text
01-ai-ready-data/lab1/explore-dataset.py
```

You will find one TODO:

- choose the target column from the CSV header

Run the script once before fixing it:

```bash
python3 01-ai-ready-data/lab1/explore-dataset.py
```

It should fail with a `NotImplementedError`. That is expected. Fix the TODO,
then run the script again.

When fixed, the script:

- loads the forest cover type CSV file
- creates a target-distribution plot
- creates distribution plots for continuous input columns
- creates target-comparison plots for continuous input columns
- creates an overview of wilderness area and soil type

The output files are numbered in the order you should inspect them.

First open:

```text
01-ai-ready-data/lab1/explore1_target_distribution.png
```

This plot shows the number of rows per forest cover type.

Questions:

- Are the class counts balanced?
- Which cover type is most common?
- Which cover types are rare?

This matters because accuracy can be misleading when one class is much more
common than another.

Then open:

```text
01-ai-ready-data/lab1/explore2_continuous_overview.png
```

This plot shows the distribution of each continuous input column. These are the
physical measurements and distances, such as elevation, slope, hillshade, and
distance to roads or water.

Questions:

- Which columns have a narrow range?
- Which columns have long tails?
- Which columns look like they might contain unusual or extreme values?

This matters because a model does not see "forest" or "terrain". It sees
numbers. You should know what those numbers look like before training.

Then open:

```text
01-ai-ready-data/lab1/explore3_continuous_vs_target.png
```

This is the most important exploration plot for the next step. It compares each
continuous input column against the target class.

Questions:

- Which columns separate the cover types most clearly?
- Does `elevation` separate some classes better than the distance columns?
- Which columns look less useful on their own?

Use this plot to choose two input columns for the next script. Do not choose at
random. The point is to make a small model from features that looked promising
before modelling.

Finally open:

```text
01-ai-ready-data/lab1/explore4_categorical_overview.png
```

This plot shows the categorical input columns: wilderness area and soil type.
They are not continuous measurements. They are category identifiers.

Questions:

- Which wilderness areas have many rows?
- Which soil types are common or rare?

After these four plots, you should know the target column, the class imbalance,
the continuous measurements, the categorical columns, and at least two candidate
features to try in the first model.

### Step 2: Train A Small Model

Open:

```text
01-ai-ready-data/lab1/train-small-model.py
```

This is a small model because it only receives two input columns. The algorithm
is still a real scikit-learn model: a random forest classifier. A random forest
trains many decision trees on slightly different views of the data, then lets
those trees vote for the final class. It is a useful first model because it can
handle non-linear patterns and mixed feature scales without much setup.

You will find two TODOs:

- use the target column you identified in the exploration step
- choose exactly two input columns after looking at the exploration graphs

Run the script once before fixing it:

```bash
python3 01-ai-ready-data/lab1/train-small-model.py
```

It should fail with a `NotImplementedError`. Fix the TODOs, then run the
script again.

When fixed, the script:

- loads the same CSV file and creates train, validation, and test splits
- trains a scikit-learn random forest classifier from only two input columns
- creates an accuracy plot
- creates a confusion-matrix plot

First open:

```text
01-ai-ready-data/lab1/small1_accuracy.png
```

The title shows the two columns you selected. The bars show the majority-class
baseline, train accuracy, validation accuracy, and test accuracy. The
majority-class baseline is the boring model that always predicts the most common
class.

Questions:

- Did your two-column model beat the majority-class reference?
- Is validation accuracy close to test accuracy?
- Did another pair of columns do better?

Try a few different pairs. Use the exploration graphs to guess which columns
will be useful, run the script, and compare the test accuracy. This is a small,
controlled version of feature selection.

Then open:

```text
01-ai-ready-data/lab1/small2_confusion_matrix.png
```

The confusion matrix shows which cover types the small model predicts correctly
and which ones it confuses. Rows are true classes. Columns are predicted
classes. A strong model has most of its counts on the diagonal.

Questions:

- Which classes are predicted well?
- Which classes are confused with each other?
- Does your two-column choice help some classes more than others?

### Step 3: Train The Big Model

Run:

```bash
python3 01-ai-ready-data/lab1/train-big-model.py
```

This script has no TODO. It uses all available input columns so you can compare
the small result with a stronger version of the same kind of model.

The big model is not a completely different machine learning idea. It is still a
random forest classifier. The important difference is the input data: instead of
asking the model to learn from two carefully chosen columns, you give it every
available feature, including elevation, distances, hillshade measurements,
wilderness area, and soil type. That lets you see how much information was left
out by the small model.

First open:

```text
01-ai-ready-data/lab1/big1_accuracy.png
```

This plot uses the same train, validation, and test split idea, but now the
model can use all input columns.

Questions:

- How much better is the big model than your best small model?
- Is the improvement small, large, or surprising?

Then open:

```text
01-ai-ready-data/lab1/big2_feature_importance.png
```

This is the model's version of the answer key. It shows which columns the
random forest used most often to make useful splits.

Questions:

- Did the important columns match your guesses from the exploration graphs?
- Is `elevation` near the top?
- Did any categorical or distance column matter more than you expected?

Finally open:

```text
01-ai-ready-data/lab1/big3_confusion_matrix.png
```

Compare this confusion matrix with the small model confusion matrix. The big
model has more information, so it should make fewer mistakes.

Questions:

- Which classes improved the most?
- Which classes are still confused?
- What does the big model prove that accuracy alone would not show?

## Lab 2: NASA Turbofan Remaining Useful Life

Lab 2 uses engine time-series data. The file is still a CSV, but one row is not
one independent example. Each engine has a sequence of cycles, and that changes
how you prepare the target and evaluate the model.

The lab has a deliberate trap. First you train a model that looks excellent.
Then you use the plots to prove that the evaluation is leaking information.
After that you train a more honest version.

Before you start the engine lab, read the dataset guide:

```text
01-ai-ready-data/lab2/dataset-guide.md
```

Keep that guide open while you work. This lab uses a few ideas that are easy to
miss if you only look at the code.

Remaining useful life means "how many cycles are left until the engine reaches
the end of its recorded life." If an engine has 200 recorded cycles, then row
cycle 1 has 199 cycles left, row cycle 100 has 100 cycles left, and row cycle
200 has 0 cycles left.

That is the model target. The dataset does not give it to you directly; you
derive it from the engine timeline.

### Step 1: Explore The Engine Trajectories

Open:

```text
01-ai-ready-data/lab2/explore-turbofan.py
```

Run:

```bash
python3 01-ai-ready-data/lab2/explore-turbofan.py
```

The script:

- loads the NASA turbofan CSV file
- counts how many cycles each engine has
- finds metric columns that actually change
- plots engine life lengths
- plots metric trajectories for three engines

First open:

```text
01-ai-ready-data/lab2/explore1_engine_cycle_counts.png
```

This is the key shift from Lab 1: the file is tabular, but the meaning is
sequential. Every row is one cycle for one engine, so the rows for the same
engine form a timeline. Each bar is one engine, and the bar height is the final
recorded cycle for that engine.

Questions:

- Do all engines have the same recorded life length?
- What does a short bar mean?
- Why can one row not tell you how close an engine is to the end?

Then open:

```text
01-ai-ready-data/lab2/explore2_metric_trajectories.png
```

When you look at the metric trajectories, focus on the shape over time. Some
signals are noisy but mostly flat. Some start drifting when an engine gets older.
That is why the target is based on where a row sits inside one engine's timeline,
not just on the row itself.

Questions:

- Which metrics clearly change over time?
- Which metrics are almost flat?
- Why might the model do better near the end of an engine's life?

### Step 2: Train A Leaky Model

Open:

```text
01-ai-ready-data/lab2/train-leaky-model.py
```

You will find one TODO:

- create the `remaining_useful_life` target

Run the script once before fixing it:

```bash
python3 01-ai-ready-data/lab2/train-leaky-model.py
```

It should fail with a `NotImplementedError`. Fix the TODO, then run it again.

The idea is:

- group rows by `engine`
- find the final recorded `cycle` for each engine
- subtract the current row's `cycle`

The code comment in the script gives you a strong hint. You are writing a small
pandas expression, not a whole new function.

Do not worry if this is your first time using `groupby`. The important idea is
that every row needs information from the other rows of the same engine. A single
row only knows its current cycle. To calculate remaining useful life, you also
need the final cycle for that engine.

The script:

- loads NASA C-MAPSS turbofan sensor data
- derives a remaining-useful-life target from each engine trajectory
- uses a random row split
- keeps `engine` as an input feature
- trains a `HistGradientBoostingRegressor`
- creates a target trajectory plot
- creates a split plot for three engines
- creates a predicted-versus-actual plot for the test rows

This version is intentionally wrong in two ways.

First, it uses a random row split. That means rows from one engine can end up in
train, validation, and test at the same time. The model has already seen part of
the test engine during training.

Second, it keeps `engine` as an input feature. `engine` is an identifier. It is
useful for grouping rows, but it is not a physical measurement. If the same
engine appears in train and test, the model can use that identifier to memorize
patterns from the workshop file.

First open:

```text
01-ai-ready-data/lab2/leaky1_target_trajectory.png
```

The target plot should show straight descending lines: as cycle increases,
remaining useful life decreases. If your plot does not look like that, your
definition of `remaining_useful_life` is probably wrong.

Questions:

- Does every example engine end at zero remaining cycles?
- Why does the line start higher for engines with a longer recorded life?
- Does this target ask for exact predictions even very early in an engine's life?

Then open:

```text
01-ai-ready-data/lab2/leaky2_split_by_engine.png
```

The split plot is the warning sign. For the same engine, you should see blue,
orange, and red points mixed together. That means rows from one engine are
spread across train, validation, and test.

Questions:

- Does one engine appear in multiple splits?
- What has the model already seen before it is tested?
- Why is this easier than predicting a completely new engine?

Finally open:

```text
01-ai-ready-data/lab2/leaky3_predictions.png
```

The prediction plot should look very strong. The text box shows the average
test error in cycles. Lower is better only when the evaluation setup is honest.

Questions:

- Are the points close to the diagonal line?
- Is the average error suspiciously good?
- Why should you not trust this result yet?

### Step 3: Train An Honest Model

Open:

```text
01-ai-ready-data/lab2/train-honest-model.py
```

You will find two TODOs:

- create the same `remaining_useful_life` target as in Step 2
- remove `engine` from the model inputs

Run the script once before fixing it:

```bash
python3 01-ai-ready-data/lab2/train-honest-model.py
```

It should fail with a `NotImplementedError`. Fix the first TODO by copying the
same target idea from the leaky model. Then fix the second TODO by removing
`engine` from the model inputs.

The split logic is already fixed for you. This script keeps whole engines
together: train engines, validation engines, and test engines are different
engines.

You still need to fix the feature list. `cycle` is allowed because it is known
while the engine is running. `engine` is not allowed because it is just an
identifier. A model that depends on engine IDs is memorizing the workshop file,
not learning a reusable maintenance signal.

Run:

```bash
python3 01-ai-ready-data/lab2/train-honest-model.py
```

First open:

```text
01-ai-ready-data/lab2/honest1_target_trajectory.png
```

The target plot should now look like the target plot from the leaky script:
straight descending lines that end at zero. The target definition is the same.
The evaluation setup is what changes.

Questions:

- Does every example engine still end at zero remaining cycles?
- Does the target definition match the one from Step 2?
- Why is it useful that the same target is used in both scripts?

Then open:

```text
01-ai-ready-data/lab2/honest2_split_by_engine.png
```

The honest split plot should show one color per engine. A train engine stays in
train. A validation engine stays in validation. A test engine stays in test.

Questions:

- Does each engine belong to exactly one split?
- Which color is the test engine?
- Why is this harder than the random row split?

Finally open:

```text
01-ai-ready-data/lab2/honest3_predictions.png
```

The honest prediction plot should look less perfect than the leaky one. That is
expected. The honest model is solving a harder and more realistic problem. The
text box again shows the average test error in cycles.

Questions:

- Is the average error higher than in the leaky model?
- Are the points more spread out?
- Is this result less flattering but more believable?

At this point, compare the leaky and honest plots:

- the leaky model has a very low average error, but train and test engines overlap
- the honest model has a higher average error, but train and test engines do not overlap

The honest result is the one you should trust.

## What You Learned

You have used two real datasets to make the first version of the problem
concrete. A CSV file can look simple, but the target column, class balance,
feature meaning, and evaluation design already shape the machine-learning
result.

You also trained small scikit-learn models. They are not toy models; models
like these are often useful in real projects. In this lab, their job is to make
the link between data, code, split design, and metrics visible. That is why the
course quickly moves beyond local files and into shared object storage.
