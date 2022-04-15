from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql import types as tp

METERS_PER_FOOT = 0.3048
FEET_PER_MILE = 5280
EARTH_RADIUS_IN_METERS = 6371e3
EARTH_RADIUS_IN_MILES = 3963.0
DISTANCE_SCALE = 2
METERS_PER_MILE = METERS_PER_FOOT * FEET_PER_MILE


def compute_distance(_spark: SparkSession, dataframe: DataFrame) -> DataFrame:
    scale_factor = fn.pow(fn.lit(10), DISTANCE_SCALE).cast(tp.DoubleType())
    return dataframe.withColumn("distance", fn.floor(fn.acos(
        fn.sin(fn.radians("start_station_latitude")) * fn.sin(fn.radians("end_station_latitude")) +
        fn.cos(fn.radians("start_station_latitude")) * fn.cos(fn.radians("end_station_latitude")) *
        fn.cos(fn.radians("start_station_longitude") - fn.radians("end_station_longitude"))
    ) * fn.lit(EARTH_RADIUS_IN_MILES) * scale_factor) / scale_factor)


def run(spark: SparkSession, input_dataset_path: str, transformed_dataset_path: str) -> None:
    input_dataset = spark.read.parquet(input_dataset_path)
    input_dataset.show()

    dataset_with_distances = compute_distance(spark, input_dataset)
    dataset_with_distances.show()

    dataset_with_distances.write.parquet(transformed_dataset_path, mode='append')
