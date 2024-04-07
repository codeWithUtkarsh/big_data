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

    # Task 5.1
    # Extract the data in January and calculate the average waiting time (use the "request_to_pickup" field) over time
    data_from_january = processable_dataframe \
        .filter(substring(col("date"), 6, 2) == "01")
    average_waiting_time_for_january = data_from_january \
        .groupBy(
            substring(col("date"), 9, 2)
                .alias("day")) \
        .agg(
            avg(col("request_to_pickup").cast(DoubleType()))
            .alias("avg_request_to_pickup"))

    average_waiting_time_for_january \
        .coalesce(1) \
        .write.option("header", True) \
        .csv("s3a://" + s3_bucket + "/bigdata/stat/average_waiting_time_for_january/")

    # Task 5.2
    # Which day(s) does the average waiting time exceed 300 seconds
    record_exceeding_300_seconds = average_waiting_time_for_january \
        .filter(col("avg_request_to_pickup") > 300)
    record_exceeding_300_seconds.show(truncate=False)

    trip_count_dataframe_by_pickup = processable_dataframe \
        .groupBy("Pickup_Borough", "time_of_day") \
        .count() \
        .withColumnRenamed("count", "trip_count") \
        .filter((col("trip_count") > 0) & (col("trip_count") < 1000))

    trip_count_dataframe_by_pickup.show(truncate=False)


    evening_trips_dataframe = processable_dataframe \
        .filter(col("time_of_day") == "evening")

    evening_trip_count_dataframe_by_pickup = evening_trips_dataframe \
        .groupBy("Pickup_Borough", "time_of_day").count()
    evening_trip_count_dataframe_by_pickup.show(truncate=False)
    
    spark.stop()
