-- 1. Query
SELECT * FROM green_taxi_11_2016 ORDER BY trip_distance DESC LIMIT 20;



-- 2. Query: Average fare amount per payment type (one-hot encoded)
SELECT
    CASE
        WHEN "payment_type_Cash" THEN 'Cash'
        WHEN "payment_type_Credit card" THEN 'Credit card'
        WHEN "payment_type_No charge" THEN 'No charge'
        WHEN "payment_type_Dispute" THEN 'Dispute'
        WHEN "payment_type_Uknown" THEN 'Unknown'
        ELSE 'Other'  
    END AS payment_type,
    AVG(fare_amount) AS average_fare_amount
FROM
    green_taxi_11_2016  
GROUP BY
    "payment_type_Cash", "payment_type_Credit card", "payment_type_No charge", "payment_type_Dispute", "payment_type_Uknown";



-- 3. Query: max Average tip amount per city
SELECT
    pu_location,
    AVG(tip_amount) AS average_tip_amount
FROM
    green_taxi_11_2016  
GROUP BY
    pu_location
ORDER BY
    average_tip_amount DESC
LIMIT
    1;



-- 4 Query: min Average tip amount per city
SELECT
    pu_location,
    AVG(tip_amount) AS average_tip_amount
FROM
    green_taxi_11_2016  
GROUP BY
    pu_location
ORDER BY
    average_tip_amount 
LIMIT
    1;



-- 5 Query: Most frequent destination based on the 'day_type' column
SELECT
    do_location,
    COUNT(*) AS trip_count
FROM
    green_taxi_11_2016  
WHERE
    day_type = 'Weekend'
GROUP BY
    do_location
ORDER BY
    trip_count DESC
LIMIT
    1;



-- 6 Query: Trip type with the highest average trip distance (one-hot encoded)
SELECT
    CASE
        WHEN "trip_type_Dispatch" THEN 'Dispatch'
        WHEN "trip_type_Street-hail" THEN 'Street-hail'
        WHEN "trip_type_Unknown" THEN 'Unknown'
        ELSE 'Other'  
    END AS trip_type,
    AVG(trip_distance) AS average_trip_distance
FROM
    green_taxi_11_2016  
GROUP BY
    "trip_type_Dispatch", "trip_type_Street-hail", "trip_type_Unknown"
ORDER BY
    average_trip_distance DESC
LIMIT 1;




-- 7 Query: Average fare amount for trips between 4 PM and 6 PM
SELECT
    AVG(fare_amount) AS average_fare_amount
FROM
    green_taxi_11_2016  
WHERE
    EXTRACT(HOUR FROM CAST("lpep_pickup_datetime" AS TIMESTAMP)) >= 16  -- 4 PM
    AND EXTRACT(HOUR FROM CAST("lpep_pickup_datetime" AS TIMESTAMP)) < 18;  -- 6 PM






