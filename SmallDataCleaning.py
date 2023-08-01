from pyspark.sql import SparkSession, functions

def main():
    house_types = [
        'Entire vacation home',
        'Entire guesthouse',
        'Entire cabin',
        'Entire villa',
        'Entire townhouse',
        'Entire cottage',
        'Entire home',
        'Entire bungalow'
    ]
    apartment_types = [
        'Entire condo',
        'Entire loft',
        'Entire rental unit',
        'Entire serviced apartment'
    ]

    file_name = [
        'Sept_Listing_Data2022',
        'Dec_Listing_Data2022',
        'March_Listing_Data2023',
        'June_Listing_Data2023'
    ]

    combined_df = None

    spark = SparkSession.builder \
        .appName("CSV Reader") \
        .getOrCreate()
    
    for filename in file_name:
        csv_file_path = filename + ".csv"
        small_df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
        columns_to_keep = ['id', 'property_type']
        small_df = small_df.select(*columns_to_keep)
        property_types_to_keep = house_types + apartment_types
        small_df = small_df.filter(functions.col('property_type').isin(property_types_to_keep))
        small_df = small_df.withColumn(
            'property_type',
            functions.when(functions.col('property_type').isin(house_types), 'House')
            .when(functions.col('property_type').isin(apartment_types), 'Apartment')
        )

        if combined_df is None:
            combined_df = small_df
        else:
            combined_df = combined_df.unionAll(small_df)
    combined_df = combined_df.distinct()   

    duplicates = combined_df.groupBy("id").count().where(functions.col("count") > 1)
    combined_df = combined_df.join(duplicates, ["id"], "left_anti")

    combined_df.coalesce(1).write.csv("property_types", header=True, mode="overwrite", compression="gzip")

    spark.stop()

if __name__ == "__main__":
    main()