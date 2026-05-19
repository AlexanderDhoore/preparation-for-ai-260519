# Chapter 7: Business Intelligence

Data preparation is not finished when a file is written.

A team also needs to see what the prepared data means.

Up to this point, most of the evidence has been technical: reports, logs,
Parquet files, Iceberg tables, Airflow runs, and model metrics. Those outputs
matter, but they are not always the best way to discuss data with a wider team.

Business Intelligence tools turn prepared datasets into SQL exploration,
charts, dashboards, and filters.

In this chapter, the main tool is Apache Superset.

Superset is not the data lake. It is not the transformation engine. It is not
where raw data should be cleaned. Superset sits near the end of the data
preparation path:

```text
object storage
  -> Iceberg table
  -> SQL query engine
  -> Superset dataset
  -> chart
  -> dashboard
  -> team discussion or decision
```

The goal is not to make a pretty screen. The goal is to make important data
questions visible.

## Why Business Intelligence Belongs Here

This course is about preparing data for AI.

That includes model training, but it also includes trust. Before a team trusts a
model, it needs to understand the data that feeds the model and the operational
world around that data.

Business Intelligence helps with questions such as:

- How many records reached the prepared table?
- Which categories dominate the dataset?
- Which values look suspicious?
- Did a metric change after a pipeline run?
- Which filters change the conclusion?
- Which numbers are safe enough to discuss with a domain expert?

Those are team questions. They often need domain experts, engineers, managers,
and researchers to look at the same numbers.

That is where a Business Intelligence tool becomes useful.

## Prepared Data, Not Raw Dumps

Superset should usually consume prepared datasets.

It can connect to many sources, but a dashboard built directly on raw files
becomes fragile. Every chart has to rediscover cleaning rules, naming rules,
and column meanings.

The cleaner pattern is:

```text
Bronze data
  -> Silver cleaned data
  -> Gold or analytical table
  -> Superset
```

In this chapter, the dataset is already available as an Iceberg table. That
means Superset does not need to know where every Parquet file lives. It only
needs a SQL connection to a query engine that can read the table.

This is why the earlier chapters matter. Object storage gives the shared
foundation. Parquet and Iceberg make analytical data queryable. Airflow can
refresh outputs. Superset makes those outputs visible to humans.

## Superset, Trino, Iceberg, And Object Storage

The workshop stack for this chapter is:

```text
Superset
  -> Trino
  -> Polaris / Iceberg catalog
  -> object storage
```

Superset is the user interface. It provides SQL Lab, chart builders,
dashboards, and filters.

Trino is the SQL query engine. It receives SQL from Superset and executes that
query against analytical tables.

Polaris is the Iceberg catalog. It tells Trino which tables exist, which
snapshots are current, and where the table metadata lives.

Object storage holds the data files.

This separation is important. Superset does not copy all lake data into its own
database. The dashboard queries stay connected to the lakehouse table. If the
table is refreshed later, the dashboard can query the refreshed data through the
same connection.

## Dimensions, Metrics, And Definitions

Business Intelligence tools usually work with dimensions and metrics.

A dimension is something you group or filter by:

- pickup hour
- payment type
- date
- product
- machine
- split
- label
- batch

A metric is something you calculate:

- trip count
- average fare
- total revenue
- average tip
- percentage of suspicious rows
- model accuracy
- average prediction error

This distinction matters because a dashboard is usually not a table dump. It is
a set of questions expressed through dimensions and metrics.

For example:

```text
show trip_count by pickup_hour
show average_total_amount by payment_type_label
show average_tip by payment_type_label
filter to payment_type_label = 'Credit card'
```

Superset makes charts easy to create, but the chart still comes from a concrete
SQL question. Before building a dashboard, it helps to run a few direct queries
and check that the columns behave the way you expect.

## SQL Lab

SQL Lab is where Superset becomes concrete.

You can inspect a queryable table before building charts:

```sql
select
    pickup_hour,
    count(*) as trip_count,
    round(avg(total_amount), 2) as average_total_amount
from business_intelligence_lab.taxi_trips
group by pickup_hour
order by pickup_hour;
```

You can also compare categories before deciding which charts are worth building:

```sql
select
    payment_type_label,
    count(*) as trip_count,
    round(avg(total_amount), 2) as average_total_amount,
    round(avg(tip_amount), 2) as average_tip_amount
from business_intelligence_lab.taxi_trips
group by payment_type_label
order by trip_count desc;
```

SQL Lab is not meant to replace Python, Spark, or Airflow. It is the last-mile
inspection layer. You use SQL to check whether a table has the shape a dashboard
needs.

SQL Lab also helps when a dashboard looks wrong. Before changing the chart, ask
whether the underlying query returns the expected rows.

## Superset Datasets

After the queryable table exists, Superset needs a dataset definition.

The dataset tells Superset:

- which table or SQL query to use
- which columns are temporal columns
- which columns are dimensions
- which metrics are available
- which columns can be filtered
- which users can access the dataset

The word "dataset" can be confusing. In Superset, a dataset is Superset's
representation of a queryable table or SQL query. It is not necessarily the same
thing as a machine-learning dataset.

You can often start with a physical table and let Superset detect the columns.
Then you refine names, descriptions, metrics, and filters.

That refinement step is important. A Superset dataset is the bridge between a
SQL table and a dashboard builder.

## Charts And Dashboards

A chart should answer one question.

Examples:

- At what hour are trips most common?
- Which payment type is most common?
- Which payment type has the highest average tip?
- Which hours have unusually expensive trips?
- Which trips have unusually high fare per mile?

In Superset, the chart builder usually asks for:

- dataset
- visualization type
- metric
- dimension or group-by column
- filters
- sort order

A dashboard collects related charts and filters.

For this chapter, a useful dashboard could contain:

- trip count by pickup hour
- average total amount by pickup hour
- payment type distribution
- average tip by payment type
- a table of high fare-per-mile trips

Dashboards should not become decoration. A useful dashboard has a decision
behind it:

```text
If this number changes, who looks at it?
What action might they take?
What context do they need before deciding?
```

## Interactivity And Filters

A static chart is useful. An interactive dashboard is usually more useful.

Superset dashboards can include native filters. A filter can apply to one chart,
several charts, or the whole dashboard.

Useful filters for the taxi dashboard include:

- pickup hour
- payment type

Filters let different users ask their own follow-up questions without changing
the underlying data pipeline.

For example, a business analyst might filter to credit card trips and inspect
tips. An operations person might filter to early morning hours and inspect
unusually expensive trips.

The technical point is simple: if you want useful filters, the analytical table
needs good dimension columns.

## What Business Intelligence Is Not

Business Intelligence is not the right place for every kind of inspection.

Superset is strong for queryable tabular data. It is not the best interface for
opening many images, listening to audio clips, inspecting video frames, or
reviewing model artifacts.

For multimedia and model evidence, other tools are better. In the next chapter,
MLflow becomes more relevant because it can track model runs, metrics, plots,
artifacts, and example outputs.

This chapter stays focused on analytical tabular data. That is enough. SQL,
charts, dashboards, and filters are a major part of how prepared data becomes
usable by a team.

## Tool Landscape

Superset is the tool you use in this workshop, but it is part of a larger
landscape.

Common commercial tools include:

- Microsoft Power BI
- Tableau
- Looker
- Qlik

Common open-source or self-hostable tools include:

- Apache Superset
- Lightdash
- Redash

Grafana is also worth knowing. It is excellent for infrastructure metrics,
logs, time-series monitoring, and operational dashboards. It can overlap with
Business Intelligence, but it is usually not the main tool for business-facing
analytical exploration.

The concepts transfer between tools: connect to prepared data, define metrics,
build charts, apply filters, share dashboards, and keep ownership clear.

## Lab 7 Preview

In the lab, you use Superset to investigate a New York taxi trip table. The
table is prepared once for the class as a shared Iceberg table, and Superset has
a shared dataset named `taxi_trips` that points to it.

The lab connects five ideas:

- prepared data should be queryable
- Superset needs a SQL query engine
- Trino can query Iceberg tables stored on object storage
- SQL Lab helps you inspect the table before building charts
- filters make dashboards useful for follow-up questions

You will not train a model in this chapter. The point is to understand how a
team can inspect prepared analytical data before it becomes input for decisions,
reports, or later machine-learning work.
