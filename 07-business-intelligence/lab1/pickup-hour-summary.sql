select
    pickup_hour,
    count(*) as trip_count,
    round(avg(total_amount), 2) as average_total_amount,
    round(avg(tip_amount), 2) as average_tip_amount,
    round(avg(trip_distance), 2) as average_distance
from business_intelligence_lab.taxi_trips
group by pickup_hour
order by pickup_hour;
