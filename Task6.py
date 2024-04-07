import sys, string
import os
import socket
import time
import operator
import boto3
import json
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql.functions import *
from graphframes import *
from pyspark.sql.types import *
from pyspark.sql.window import *


if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("Task_1")\
        .getOrCreate()
    
    s3_data_repository_bucket = os.environ['DATA_REPOSITORY_BUCKET']

    s3_endpoint_url = os.environ['S3_ENDPOINT_URL']+':'+os.environ['BUCKET_PORT']
    s3_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    s3_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    s3_bucket = os.environ['BUCKET_NAME']

    hadoopConf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoopConf.set("fs.s3a.endpoint", s3_endpoint_url)
    hadoopConf.set("fs.s3a.access.key", s3_access_key_id)
    hadoopConf.set("fs.s3a.secret.key", s3_secret_access_key)
    hadoopConf.set("fs.s3a.path.style.access", "true")
    hadoopConf.set("fs.s3a.connection.ssl.enabled", "false")

    rideshare_data_csv_path = '/ECS765/rideshare_2023/rideshare_data.csv'
    taxi_zone_lookup_csv_path = '/ECS765/rideshare_2023/taxi_zone_lookup.csv'
    
    rideshare_data_from_csv = spark \
        .read.csv("s3a://" + s3_data_repository_bucket + rideshare_data_csv_path, header=True) \
        .repartition(600)
    taxi_zone_lookup_data_from_csv = spark \
        .read.csv("s3a://" + s3_data_repository_bucket + taxi_zone_lookup_csv_path, header=True) \
        .repartition(200)
    
    first_join_on_pickup_data = rideshare_data_from_csv \
        .join(
            broadcast(taxi_zone_lookup_data_from_csv), 
            col("pickup_location") == col("LocationID"), 
            "left_outer") \
    .withColumnRenamed("Borough", "Pickup_Borough") \
    .withColumnRenamed("Zone", "Pickup_Zone") \
    .withColumnRenamed("service_zone", "Pickup_service_zone") \
    .drop("LocationID")

    second_join_on_dropoff = first_join_on_pickup_data \
        .join(
            broadcast(taxi_zone_lookup_data_from_csv), 
            col("dropoff_location") == col("LocationID"), 
            "left_outer") \
    .withColumnRenamed("Borough", "Dropoff_Borough") \
    .withColumnRenamed("Zone", "Dropoff_Zone") \
    .withColumnRenamed("service_zone", "Dropoff_service_zone") \
    .drop("LocationID")
    
    second_join_on_dropoff_after_converting = second_join_on_dropoff \
        .withColumn("date", from_unixtime(col("date").cast("bigint"), "yyyy-MM-dd"))

    processable_dataframe = second_join_on_dropoff_after_converting \
        .withColumn("month", substring(col("date"), 6, 2))
    processable_dataframe = processable_dataframe \
        .withColumn("business_month", concat(col("business"), lit("-"), col("month")))

    data_from_january = processable_dataframe \
        .filter(substring(col("date"), 6, 2) == "01")
    average_waiting_time_for_january = data_from_january \
        .groupBy(
            substring(col("date"), 9, 2)
                .alias("day")) \
        .agg(
            avg(col("request_to_pickup").cast(DoubleType()))
            .alias("avg_request_to_pickup"))

    # Task 6.1
    # Find trip counts greater than 0 and less than 1000 for different 'Pickup_Borough' at different 'time_of_day'
    trip_count_dataframe_by_pickup = processable_dataframe \
        .groupBy("Pickup_Borough", "time_of_day") \
        .count() \
        .withColumnRenamed("count", "trip_count") \
        .filter((col("trip_count") > 0) & (col("trip_count") < 1000))

    trip_count_dataframe_by_pickup.show(truncate=False)

    # Task 6.2
    # Calculate the number of trips for each 'Pickup_Borough' in the evening time (i.e., time_of_day field)
    evening_trips_dataframe = processable_dataframe
        .filter(col("time_of_day") == "evening")

    evening_trip_count_dataframe_by_pickup = evening_trips_dataframe \
        .groupBy("Pickup_Borough", "time_of_day").count()
    evening_trip_count_dataframe_by_pickup.show(truncate=False)
    
    # Task 6.3
    # Calculate the number of trips that started in Brooklyn (Pickup_Borough field) and ended in Staten Island (Dropoff_Borough field)
    brooklyn_to_staten_island_records = processable_dataframe \
        .filter(
            (col("Pickup_Borough") == "Brooklyn") & (col("Dropoff_Borough") == "Staten Island"))
    brooklyn_to_staten_island_records \
        .select("Pickup_Borough", "Dropoff_Borough", "Pickup_Zone") \
        .show(10, truncate=False)

    count = brooklyn_to_staten_island_records.count()
    print("--------- Total number of trips from Brooklyn to Staten Island------------", count)
    
    spark.stop()
