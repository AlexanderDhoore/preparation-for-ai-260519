# Lab 7: Business Intelligence

In this lab you use Apache Superset to explore a prepared New York taxi trip
table.

You will use:

- Superset as the Business Intelligence user interface
- Trino as the SQL query engine
- an Iceberg table stored on object storage

The challenge is practical: use SQL first to understand the table, then use the
Superset chart builder to turn a few useful questions into charts and a small
dashboard.

## Platform Shape

The table for this lab is:

```text
business_intelligence_lab.taxi_trips
```

Superset does not store the taxi rows itself. The path is:

```text
Superset
  -> Course Trino database connection
  -> business_intelligence_lab.taxi_trips Iceberg table
  -> object storage
```

That means Superset is the exploration layer. Trino executes the SQL. Iceberg
tracks the table. Object storage holds the Parquet data files.

Important columns:

- `pickup_at`: timestamp when the trip started
- `pickup_date`: date extracted from the pickup timestamp
- `pickup_hour`: hour extracted from the pickup timestamp
- `passenger_count`: number of passengers
- `trip_distance`: trip distance in miles
- `payment_type_label`: readable payment type
- `fare_amount`: fare before extras
- `tip_amount`: tip amount
- `total_amount`: total charged amount
- `trip_minutes`: trip duration in minutes
- `fare_per_mile`: fare amount divided by trip distance

## Log In To Superset

Your VM already has your Superset account in `participant.env`:

```bash
grep '^SUPERSET_' participant.env
```

Open the URL from `SUPERSET_URL`, then log in with `SUPERSET_USERNAME` and
`SUPERSET_PASSWORD`.

For the workshop platform, that URL is:

```text
http://superset.mechatronics.lan
```

After login, the top navigation should contain:

- `Dashboards`
- `Charts`
- `Datasets`
- `SQL`

Superset is a shared service. Use names that include your participant id, for
example `preparation-for-ai-XX`, when you save charts or dashboards.

The main Superset objects fit together like this:

```text
SQL Lab
  -> run one-off SQL queries

Dataset
  -> saved connection to a table or query
  -> used as the source for charts

Chart
  -> one visual question from a dataset

Dashboard
  -> several charts plus shared filters
```

## Step 1: Inspect Rows In SQL Lab

In the top navigation, open:

```text
SQL -> SQL Lab
```

The SQL menu opens a dropdown. Click `SQL Lab`.

Use the `Course Trino` database. You can leave the schema selector empty if you
use the fully qualified table name in your SQL.

Open this file:

```text
07-business-intelligence/lab1/inspect-table.sql
```

Paste the query into SQL Lab and click `Run`.

You should see 20 rows from the taxi table.

Look for:

- timestamps such as `pickup_at` and `dropoff_at`
- analytical dimensions such as `pickup_hour` and `payment_type_label`
- numeric measurements such as `trip_distance`, `tip_amount`, and
  `total_amount`

This is your first sanity check. Before building charts, make sure the table
has the columns a dashboard needs.

## Step 2: Run A Few SQL Questions

Before creating charts, use SQL Lab to ask a few direct questions.

Open this file:

```text
07-business-intelligence/lab1/payment-type-summary.sql
```

Paste it into SQL Lab and click `Run`.

This query groups trips by payment type. It shows trip count, average total
amount, average tip amount, and average distance.

Look for:

- which payment type dominates the dataset
- whether tip amounts behave differently by payment type
- whether the average trip distance differs by payment type

Then open:

```text
07-business-intelligence/lab1/pickup-hour-summary.sql
```

Run that query too.

This query groups trips by pickup hour. It is a good bridge from SQL to charts:
if a result has one row per hour, it will probably make a readable bar chart.

Use the SQL results to decide which charts are worth building.

## Step 3: Open The Superset Dataset

In the top navigation, click:

```text
Datasets
```

Search for:

```text
taxi_trips
```

Click the `taxi_trips` dataset name.

This opens the chart builder. You should see:

- chart title field at the top
- `Chart Source` showing `business_intelligence_lab.taxi_trips`
- a left-side list of metrics and columns
- the `Data` tab with chart controls
- the `Create chart` button

The dataset is Superset's representation of the Iceberg table. It is not a copy
of the data.

## Step 4: Create A Trip Count Chart

The chart builder has a column and metric list on the left, chart controls in
the middle, and the chart output on the right.

1. Enter a chart title such as:

```text
Trips By Pickup Hour - preparation-for-ai-XX
```

2. In the chart type list, click `Bar Chart`.
3. In the `Data` tab, find `X-axis`.
4. Drag `pickup_hour` from the left `Columns` list into `X-axis`.
5. Drag `COUNT(*)` from the left `Metrics` list into `Metrics`.
6. Click `Create chart`.

You should see a bar chart with one bar per pickup hour.

Questions to check in the chart:

- Which pickup hours have the most trips?
- Does the shape look like a normal daily rhythm?
- Would a table or a chart be easier for this question?

Click `Save`. Save it as a new chart. You can create a new dashboard from the
save dialog or add it to a dashboard later.

## Step 5: Create A Payment Type Chart

Go back to `Datasets`, search for `taxi_trips`, and click the dataset again.

Build another chart:

- title: `Trips By Payment Type - preparation-for-ai-XX`
- chart type: `Bar Chart`
- `X-axis`: drag `payment_type_label`
- `Metrics`: drag `COUNT(*)`

Click `Create chart`.

This chart should make one thing obvious: one payment type dominates the
prepared sample.

Save the chart.

## Step 6: Create An Investigation Table

Open this SQL file:

```text
07-business-intelligence/lab1/high-fare-per-mile.sql
```

Run it in SQL Lab.

This query is not a dashboard summary. It is an investigation query. It shows
trips with the highest fare per mile.

Use the result to answer:

- Do these rows look like normal trips?
- Which columns help you decide whether they are suspicious?
- Would this be better as a dashboard chart or as a table you inspect directly?

You can also make this query into a table chart from SQL Lab by clicking
`Create Chart` after the query succeeds.

## Step 7: Build A Dashboard

Open:

```text
Dashboards
```

Click `+ Dashboard` and create a dashboard named:

```text
Taxi Analytics - preparation-for-ai-XX
```

Add your saved charts.

Then add at least one dashboard filter. Good filter columns are:

- `pickup_hour`
- `payment_type_label`

Use the filter and check whether the charts update together.

That interaction is the reason to use a dashboard instead of separate static
images. A dashboard lets someone ask a follow-up question without editing SQL.

## Step 8: Investigate One Question

Use your dashboard to investigate this question:

```text
When are taxi trips most common, and which payment types behave differently?
```

Do not write a report. Use the dashboard itself as the answer.

Check:

- whether trip volume changes by hour
- whether credit card and cash trips behave differently
- whether dashboard filters help you explore the table faster than SQL alone
- whether the high fare-per-mile table reveals rows that deserve closer
  inspection

## What You Should Notice

Superset works best when the data is already prepared and queryable.

The table in this lab is stored in object storage as an Iceberg table. Superset
does not query files directly. Superset sends SQL to Trino, Trino reads the
Iceberg table, and the dashboard receives the result.

SQL Lab and dashboards serve different purposes. SQL Lab is good for direct
inspection. Charts and dashboards are better when you want a reusable visual
view that other people can filter and explore.
