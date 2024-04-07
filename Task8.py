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
    
    # Task 8.1
    # Define the StructType of vertexSchema and edgeSchema
    vertex_schema = StructType([
        StructField("LocationID", IntegerType(), nullable=False),
        StructField("Borough", StringType(), nullable=True),
        StructField("Zone", StringType(), nullable=True),
        StructField("service_zone", StringType(), nullable=True)])
    
    edge_schema = StructType([
        StructField("src", IntegerType(), nullable=False),
        StructField("dst", IntegerType(), nullable=False)])

    # Task 8.2
    dataframe_for_vertices = spark \
        .read \
        .csv("s3a://" + s3_data_repository_bucket + taxi_zone_lookup_csv_path, header=True, schema=vertex_schema) \
        .repartition(200)
    
    dataframe_for_edges = spark \
        .read \
        .csv("s3a://" + s3_data_repository_bucket + rideshare_data_csv_path, header=True) \
        .select(col("pickup_location").alias("src"), col("dropoff_location").alias("dst")) \
        .repartition(600)

    dataframe_for_vertices = dataframe_for_vertices.withColumnRenamed('LocationID', 'id')

    # Show data dataframe_for_vertices & dataframe_for_edges 
    dataframe_for_vertices.show(truncate=False)
    dataframe_for_edges.show(truncate=False)

    # Task 8.3
    # Creating graph from vertices and edges and then creating dataframe for the traversed path (source to destination)
    curated_graph = GraphFrame(dataframe_for_vertices, dataframe_for_edges)
    dataframe_for_graph_traversal = curated_graph \
        .find("(src)-[edge]->(dst)")

    # Show data dataframe_for_graph_traversal  
    dataframe_for_graph_traversal.select("src", "edge", "dst").show(truncate=False, n=10)

    # Task 8.4
    #Count connected vertices with the same Borough and same service_zone
    dataframe_for_connected__path_with_counts = curated_graph.edges \
        .join( \
            dataframe_for_vertices \
                .withColumnRenamed("Borough", "src_Borough") \
                .withColumnRenamed("service_zone", "src_service_zone"), curated_graph.edges.src == dataframe_for_vertices.id, 'left') \
        .join( \
            dataframe_for_vertices \
                .withColumnRenamed("Borough", "dst_Borough")
                .withColumnRenamed("service_zone", "dst_service_zone"), curated_graph.edges.dst == dataframe_for_vertices.id, 'left') \
        .filter( \
            (col("src_Borough") == col("dst_Borough")) & (col("src_service_zone") == col("dst_service_zone"))) \
        .select(
            col("src").alias("id"), 
            col("dst").alias("id"), 
            col("src_Borough").alias("Borough"), 
            col("src_service_zone").alias("service_zone"))

    dataframe_for_connected__path_with_counts.show(10, truncate=False)

    # Task 8.5
    # perform page ranking on the graph dataframe.
    dataframe_for_page_rank = curated_graph \
        .pageRank(resetProbability=0.17, tol=0.01) \
        .vertices
    dataframe_for_page_rank = dataframe_for_page_rank \
        .orderBy(col("pagerank").desc())

    # Show data dataframe_for_page_rank  
    dataframe_for_page_rank.select("id", "pagerank").show(5, truncate=False)

    spark.stop()
