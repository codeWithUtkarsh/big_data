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

    # Task 2.1 Count the number of trips for each business in each month. 
    # Draw the histogram with 'business-month' on the x-axis and trip counts on the y-axis
    calculated_trip_count = processable_dataframe \
        .groupBy("business_month") \
        .count() \
        .orderBy("business_month")

    calculated_trip_count.show()
    
    # Writing calculated_trip_count dataframe in csv
    calculated_trip_count \
        .coalesce(1) \
        .write.option("header", True) \
        .csv("s3a://" + s3_bucket + "/data/stat/calculated_trip_count/")

    # Task 2.2 Calculate the platform's profits (rideshare_profit field) for each business in each month. 
    # Draw the histogram with 'business-month' on the x-axis and the platform's profits on the y-axis. 
    calculated_profits = processable_dataframe \
        .groupBy("business_month") \
        .agg({"rideshare_profit": "sum"}) \
        .orderBy("business_month")

    calculated_profits.show()
    # Writing calculated_profits dataframe in csv
    calculated_profits \
        .coalesce(1) \
        .write.option("header", True) \
        .csv("s3a://" + s3_bucket + "/data/stat/calculated_profits/")

    # Task 2.3 Calculate the driver's earnings (driver_total_pay field) for each business in each month. 
    # Draw the histogram with 'business-month' on the x-axis and the driver's earnings on the y-axis 
    calculated_earnings = processable_dataframe \
        .groupBy("business_month") \
        .agg({"driver_total_pay": "sum"}) \
        .orderBy("business_month")

    calculated_earnings.show()
    # Writing calculated_earnings dataframe in csv
    calculated_earnings \
        .coalesce(1) \
        .write.option("header", True) \
        .csv("s3a://" + s3_bucket + "/data/stat/calculated_earnings/")
    
    spark.stop()
