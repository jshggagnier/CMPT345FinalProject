import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types

ListingValue_schema = types.StructType([
    types.StructField('listing_id', types.LongType()),
    types.StructField('date', types.DateType()),
    types.StructField('price', types.StringType()),
    types.StructField('adjusted_price', types.StringType()),
])

ListingType_schema = types.StructType([
    types.StructField('listing_id', types.LongType()),
    types.StructField('Type', types.StringType()),
])

def main(in_directory, in_directory2, out_directory):
    
    spark = SparkSession.builder.appName('Reddit Relative Scores').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')
    
    PropertyTypes = spark.read.csv(in_directory, schema=ListingType_schema,header=True)
    Values = spark.read.csv(in_directory2, schema=ListingValue_schema,header=True)

    JoinedData = Values.join(PropertyTypes,on="listing_id",how="inner").select(["date","type","price"])
    JoinedData = JoinedData.groupby("date").pivot("Type",["Apartment","House"]).agg({"price":"avg"})
    JoinedData = JoinedData.sort("date")

    #safe to do as the single computer we are using already has all the files, and this is simply less than there was before
    JoinedData.coalesce(1).write.parquet(out_directory, mode='overwrite',compression="gzip")


if __name__=='__main__':
    in_directory = "property_types"
    in_directory2 = "AirBNB-Data-PysparkFiltered"
    out_directory = "AirBNB-By-Category"
   

    main(in_directory, in_directory2, out_directory)
