import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types


ListingValue_schema = types.StructType([
    types.StructField('listing_id', types.LongType()),
    types.StructField('date', types.DateType()),
    types.StructField('available', types.StringType()),
    types.StructField('price', types.StringType()),
    types.StructField('adjusted_price', types.StringType()),
    types.StructField('minimum_nights', types.LongType()),
    types.StructField('maximum_nights', types.LongType()),
])

def main(in_directory, out_directory):
    values = spark.read.csv(in_directory, schema=ListingValue_schema,header=True)
    values.show(10)
    values = values.filter(values["available"]=="f")
    values.show(10)
    values = values.withColumn("cleanedprice",functions.regexp_replace(values["price"],r"[\$#,]","").astype(types.LongType()))
    values = values.withColumn("cleanedpriceadj",functions.regexp_replace(values["adjusted_price"],r"[\$#,]","").astype(types.LongType())).select(["listing_id","date","cleanedprice","cleanedpriceadj"])


    #values = values.groupby("listing_id").agg(functions.avg("cleanedprice"),functions.avg("cleanedpriceadj"),functions.min("date"),functions.max("date"))
    #values.corr("avg(cleanedprice)","avg(cleanedpriceadj)")
    #Averages (Cache this)

    # Sort by's (these can be shared and done locally)

    values.coalesce(1).write.csv(out_directory, header=True, mode='overwrite',compression="gzip")


if __name__=='__main__':
    in_directory = "AirBNB-Data-cleaned"
    out_directory = "output"
    spark = SparkSession.builder.appName('Reddit Relative Scores').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)
