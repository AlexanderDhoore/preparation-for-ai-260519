# Forest Cover Type Dataset Guide

This lab uses the public Hugging Face dataset `mstz/covertype`.

Each row describes a 30 by 30 meter patch of forest land. The target column is
`cover_type`, which describes the dominant tree cover type for that patch.

The target values mean:

- `0`: Spruce/Fir
- `1`: Lodgepole Pine
- `2`: Ponderosa Pine
- `3`: Cottonwood/Willow
- `4`: Aspen
- `5`: Douglas-fir
- `6`: Krummholz

This dataset is much larger than the first tiny datasets people often see in
introductory machine learning. It also has seven target classes, which makes the
confusion matrix more interesting than a simple yes/no classifier.

## Continuous Cartographic Columns

### elevation

Elevation of the land patch. This is usually one of the strongest signals,
because different tree species prefer different altitude ranges.

### aspect

Direction that the slope faces, measured in degrees. Aspect can influence sun
exposure, temperature, and moisture.

### slope

Steepness of the terrain.

### horizontal_distance_to_hydrology

Horizontal distance to the nearest surface water feature.

### vertical_distance_to_hydrology

Vertical distance to the nearest surface water feature.

### horizontal_distance_to_roadways

Horizontal distance to the nearest roadway. This can indirectly capture human
access, terrain type, and location.

### hillshade_9am, hillshade_noon, hillshade_3pm

Estimated hillshade at different times of day. These columns describe how much
light reaches the patch based on terrain.

### horizontal_distance_to_fire_points

Horizontal distance to known wildfire ignition points.

## Categorical Columns

### wilderness_area

Wilderness area category. This tells you which protected area the patch belongs to.

### soil_type

Soil type category. Soil influences moisture, nutrients, and which trees can grow well.

## What To Watch For

Before trusting a model, ask:

- Which cover type is most common?
- Which cover types are rare?
- Are rare classes harder for the model to learn?
- Which terrain measurements separate classes visually?
- Which classes are confused with each other?
- Do the most important features make ecological sense?
