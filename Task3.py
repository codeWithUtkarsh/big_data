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

    # Task 3.1
    # Identify the top 5 popular pickup boroughs each month
    dataframe_grouped_by_pickup = processable_dataframe \
        .groupBy("Pickup_Borough", "month") \
        .count()
    window_criteria = Window \
        .partitionBy("month") \
        .orderBy(col("count").desc())
    dataframe_ranked_over_window = dataframe_grouped_by_pickup \
        .withColumn("rank", rank().over(window_criteria))
    
    top_5_record = dataframe_ranked_over_window \
        .filter(col("rank") <= 5)
    selected_dataframe = top_5_record \
        .select(
            "Pickup_Borough", 
            "month", 
            col("count").alias("trip_count"))
    
    selected_dataframe.show(truncate=False, n=25)

    # Task 3.2
    dataframe_grouped_by_pickup = processable_dataframe \
        .groupBy("Dropoff_Borough", "month").count()
    dataframe_ranked_over_window = dataframe_grouped_by_pickup \
        .withColumn("rank", rank().over(window_criteria))
    
    top_5_record = dataframe_ranked_over_window.filter(col("rank") <= 5)
    selected_dataframe = top_5_record \
        .select(
            "Dropoff_Borough", 
            "month", 
            col("count").alias("trip_count"))
    selected_dataframe.show(n=25, truncate=False)

    # Task 3.3
    calculated_route_profit = processable_dataframe \
        .select(
            concat(
                col("Pickup_Borough"), 
                lit(" to "), 
                col("Dropoff_Borough"))
            .alias("Route"),  
            "driver_total_pay") \
                .groupBy("Route") \
                .agg(
                    sum(col("driver_total_pay").cast(DoubleType()))
                        .alias("total_profit")) \
                .orderBy(col("total_profit").desc())
    calculated_route_profit.show(truncate=False, n=30)

    spark.stop()
