select
    pickup_at,
    payment_type_label,
    trip_distance,
    trip_minutes,
    fare_amount,
    total_amount,
    fare_per_mile
from business_intelligence_lab.taxi_trips
order by fare_per_mile desc
limit 20;
