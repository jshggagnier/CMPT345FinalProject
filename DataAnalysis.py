import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

## data frame contains 221 months worth of average prices for houses of varying types, first column is date
monthlyAvgPrice = pd.read_csv("ByApartment.csv", parse_dates=['Date'])

#These data sets are the stats for the listings (may need to be accessed in spark)
firstQuarterListings = pd.read_csv("Sept_Listing_Data2022.csv")
secondQuarterListings = pd.read_csv("Dec_Listing_Data2022.csv")
thirdQuarterListings = pd.read_csv("March_Listing_Data2023.csv")
fourthQuarterListings = pd.read_csv("June_Listing_Data2023.csv")

## LargeAIRBNBCalendar contains the daily prices of the listings, but doesn't have anything other than the ID associated
## join the listings (Id,Property_type) datasets with the summarised monthly prices (ID, Average price for the month/day)
## We can summarise by month or day if we want. This probably should be done in spark, but feel free to use any method that you know.


# Sample plot to ensure you have both python and the data downloaded.
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Apartment_unit"],c="blue",label="apartments")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Two_storey"],c="red",label="two-storey")
plot.title("comparison of different forms of housing")
plot.ylabel("cost ($)")
plot.xlabel("year of sale")
plot.legend()
plot.show()

print("Hello world! Analysis complete!")