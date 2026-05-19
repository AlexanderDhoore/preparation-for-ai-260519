# NASA Turbofan Dataset Guide

This lab uses the public Hugging Face dataset
`nominal-io/nasa-turbofan-degradation`.

The data comes from the NASA C-MAPSS turbofan degradation simulation. Each row
describes one engine at one operating cycle. An engine appears many times: once
for every cycle in its recorded life.

## What One Row Means

One row is not one independent machine. One row is one moment in the life of one
engine.

The important identifier columns are:

- `engine`: which engine the row belongs to
- `cycle`: the cycle number for that engine

That means the dataset is tabular in storage format, but sequential in meaning.
Rows from the same engine are connected.

## Sensor Columns

The dataset contains operating settings and sensor measurements. Examples:

- `setting_1`
- `setting_2`
- `setting_3`
- `lpc_outlet_temperature_r`
- `hpc_outlet_temperature_r`
- `lpt_outlet_temperature_r`
- `physical_core_speed_rpm`
- `hpc_outlet_static_pressure_psia`
- `ratio_of_fuel_flow_to_ps30_pps_psia`
- `corrected_core_speed_rpm`
- `high_pressure_turbines_cool_air_flow`
- `low_pressure_turbines_cool_air_flow`

Some columns barely change. Some columns change visibly as the engine gets
closer to the end of its recorded life. The exploration script helps you see
that before you train a model.

## Target Column

The original file does not contain a ready-made target column for this lab.
You create it.

For each engine:

- find the last recorded cycle
- subtract the current cycle

That gives `remaining_useful_life`.

Example:

```text
engine 1 has maximum cycle 192
at cycle 1, remaining useful life is 191
at cycle 100, remaining useful life is 92
at cycle 192, remaining useful life is 0
```

This target is sometimes called a regression target. Regression means the model
predicts a number instead of a class label. In Lab 1, the model predicted a
forest cover type. In this lab, the model predicts a number of cycles.

## Train, Validation, And Test Splits

The split matters because rows from the same engine are connected.

A random row split can put cycle 10 from engine 7 in training and cycle 100 from
engine 7 in testing. That looks like a normal split, but it leaks information.
The model is being tested on an engine it has already partly seen.

An engine split keeps whole engines together. If engine 7 is in training, no row
from engine 7 appears in validation or test. That is harder, but it matches the
real question better: can the model work on an engine it has not seen before?
