# Lab 4: Open Table Formats

In Chapter 3 you worked with Parquet files directly. In this lab you add an
open table format on top.

You will use public NYC Yellow Taxi trip data. The source files are monthly
Parquet files in the shared object-storage bucket. Each source month in the
public dataset has about 3 million rows and is roughly 48 MiB as compressed
Parquet. For the lab, the instructor prepares a cleaned random sample of
1 million rows per month and uploads that to the shared bucket. That shared
Bronze dataset is read-only for students.

The table you create is different. It will be an Iceberg table in your own
participant namespace and your own participant bucket. That table is yours to
write to: you can append data, update table metadata, inspect snapshots, and
later use engines that support Iceberg mutations.

The main idea is interoperability:

```text
PyIceberg creates the table from Python.
DuckDB queries the same table with SQL.
Spark appends another month.
```

The storage stays shared and open, but the dataset now has a table name,
schemas, snapshots, and commit history.

## Platform Shape

The workshop has one shared Polaris catalog service. Your URL and credentials
are configured in `participant.env`.

Keep the two storage roles separate in your head:

```text
shared bucket:
  read-only monthly NYC taxi Parquet samples prepared by the instructor

your participant bucket and Iceberg namespace:
  read/write Iceberg table built from those source files
```

Your Iceberg namespace is SQL-friendly. For numbered participants, it looks
like:

```text
preparation_for_ai_01
preparation_for_ai_02
```

The table name used in this lab is:

```text
yellow_taxi_trips
```

The full table name therefore looks like this:

```text
course.preparation_for_ai_XX.yellow_taxi_trips
```

`course` is the local catalog alias used by the query engine.
`preparation_for_ai_XX` is your namespace in Polaris. `yellow_taxi_trips` is
the table.

## Start In The Course Environment

Run:

```bash
source .venv/bin/activate
```

Also check that your Kubernetes credential file exists:

```bash
ls -l participant.kubeconfig
```

## Step 1: Create An Iceberg Table With PyIceberg

Open:

```text
04-open-table-formats/lab1/pyiceberg-table.py
```

The script contains two TODOs.

First, use the table name:

```text
yellow_taxi_trips
```

Second, find the `table.append(...)` call. This is the PyIceberg operation that
adds data to the table and creates a new table snapshot. Replace the placeholder
inside that call with the Arrow table variable that was loaded from the shared
taxi Parquet file.

Then run:

```bash
python3 04-open-table-formats/lab1/pyiceberg-table.py
```

Expected local outputs:

```text
04-open-table-formats/lab1/pyiceberg1_created_table.png
```

The script reads the prepared January 2024 taxi Parquet sample from the shared
bucket, creates your Iceberg namespace if needed, and creates the Iceberg table.

Open:

```text
04-open-table-formats/lab1/pyiceberg1_created_table.png
```

This plot checks the basic table creation. The rows written and rows read back
should match. That tells you PyIceberg did not only write files; it also created
a table that can be read through the Iceberg metadata. The bottom of the plot
also shows the current snapshot ID and the table columns.

The important shift is that the table is no longer only an S3 path. It is now a
named table in a catalog.

Keep the current snapshot ID visible. You will compare it with the table state
after Spark appends another month. That snapshot ID is the proof that the table
has a versioned state, not only a folder of files.

This is how Iceberg makes a table feel mutable on top of object storage. The
Parquet files themselves are still ordinary objects. When a writer appends data,
it writes new files and commits new metadata. The catalog then points the table
name to a new snapshot. Engines such as PyIceberg, DuckDB, and Spark follow that
snapshot pointer, so the table can change while the table name stays stable.

## Step 2: Query The Same Table With DuckDB

Open:

```text
04-open-table-formats/lab1/duckdb-iceberg.py
```

Complete the two small TODOs:

- use the same table name again
- fix the row-count expression in the payment-type SQL query

For the table name, use:

```text
yellow_taxi_trips
```

For the SQL query, scroll to the `SELECT` statement that groups by
`payment_type`. The missing expression should count rows per payment type and
name that result `rows`, because the query orders by that value.

Then run:

```bash
python3 04-open-table-formats/lab1/duckdb-iceberg.py
```

Expected local outputs:

```text
04-open-table-formats/lab1/duckdb1_column_ranges.png
04-open-table-formats/lab1/duckdb2_daily_overview.png
04-open-table-formats/lab1/duckdb3_payment_type_overview.png
```

DuckDB does not recreate the table. It attaches to Polaris, resolves the table
name, and queries the table that PyIceberg created.

That is the first Iceberg moment in this lab: Python created the table, but SQL
can query it without changing the data layout.

The terminal output also prints the current snapshot ID. It should match the
snapshot ID from the PyIceberg plot, because DuckDB is still looking at the same
table version.

First open:

```text
04-open-table-formats/lab1/duckdb1_column_ranges.png
```

This image is the quick dataset tour. Each small plot shows how many rows fall
into a range or category for one column. Use it to get a feel for the taxi data
before reading the more specific summaries. Look for columns with narrow ranges,
long tails, and category values that dominate the table.

Then open:

```text
04-open-table-formats/lab1/duckdb2_daily_overview.png
```

This plot groups the same table by pickup day. It shows row count, average trip
distance, average total amount, and average tip amount. At this point it should
show January days. After Spark appends February, the same plot will visibly
expand.

Then open:

```text
04-open-table-formats/lab1/duckdb3_payment_type_overview.png
```

This plot groups by payment type and shows the same style of summary from a
different angle. The axis labels translate the numeric taxi payment codes into
names such as credit card, cash, dispute, and no charge. DuckDB did not receive
a list of Parquet files. It attached to Polaris, resolved the table name, and
queried the Iceberg table created by PyIceberg.

Questions:

- Which columns have long tails or uneven category counts?
- Which day has the most trips?
- Do trips, totals, and tips move together by day?
- Which payment type has the most trips?
- Which payment type has the highest average total amount?
- Why is it useful that DuckDB can query the table by name?

This is the interoperability check. If two engines see the same table state,
the catalog is doing useful work.

## Step 3: Append Another Month With Spark

Open:

```text
04-open-table-formats/lab1/spark-iceberg.py
```

Use the same table name again.

Then generate and apply the Spark job:

```bash
python3 04-open-table-formats/lab1/spark-iceberg.py
KUBECONFIG=participant.kubeconfig kubectl apply -f 04-open-table-formats/lab1/spark_iceberg_table_application.yaml
```

Watch it:

```bash
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX get sparkapp chapter04-iceberg-spark -w
```

Inspect it:

```bash
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX describe sparkapp chapter04-iceberg-spark
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX get pods
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX logs <driver-pod-name> --tail=100
```

Spark reads the prepared February 2024 taxi Parquet sample from the shared
bucket and appends those records to the same Iceberg table. It does not write
extra report files. The point is deliberately simple: Spark changes the table,
then DuckDB should see that changed table state.

In the Spark driver logs, look for:

```text
Rows before: 1000000
Appended February rows: 1000000
Rows after: 2000000
```

Spark created a new committed table state when it appended February. It did not
edit the January Parquet files in place. It added February data files and moved
the table forward to a new snapshot. You will confirm that new snapshot with
DuckDB in the next step.

After Spark finishes, run Step 2 again:

```bash
python3 04-open-table-formats/lab1/duckdb-iceberg.py
```

The row count should now include both January and February. That is the second
Iceberg moment: Spark changed the table, and DuckDB sees the new table state by
using the same catalog name. The DuckDB terminal output should also print the
new snapshot ID.

After rerunning DuckDB, reopen:

```text
04-open-table-formats/lab1/duckdb2_daily_overview.png
```

The plot should now show January and February days. That visual change is the
main evidence that Spark updated the Iceberg table and DuckDB saw the new table
state through the same catalog name.

Questions:

- Did the row count increase?
- Did the snapshot ID change?
- Can DuckDB query the table without knowing which Parquet files Spark wrote?

That comparison is the lab's main evidence. The table changed, but the table
name stayed stable.

## Clean Up Spark Applications

The Spark job keeps the driver pod around so you can inspect logs. When you are
done, clean up your application and ConfigMap:

```bash
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX delete sparkapp chapter04-iceberg-spark
KUBECONFIG=participant.kubeconfig kubectl -n preparation-for-ai-XX delete configmap chapter04-iceberg-spark-code
```

## What You Learned

You started with monthly Parquet files in object storage.

Then you added:

- a shared catalog
- a named Iceberg table
- snapshots
- a Spark append job

Object storage still holds the data, but the platform now has table state.

That table state is what makes the dataset usable by multiple engines and
reproducible model runs. A future model should not only say "I trained from an
S3 prefix." It should be able to point to a table name and snapshot.
