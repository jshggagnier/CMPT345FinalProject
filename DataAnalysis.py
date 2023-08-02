import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sys
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency

## data frame contains 221 months worth of average prices for houses of varying types, first column is date
monthlyAvgPrice = pd.read_csv("ByApartment.csv", parse_dates=['Date'])

# data column
composite_prices = monthlyAvgPrice['Composite']
apartment_unit_prices = monthlyAvgPrice['Apartment_unit']
one_storey_prices = monthlyAvgPrice['One_storey']
two_storey_prices = monthlyAvgPrice['Two_storey']
townhouse_prices = monthlyAvgPrice ['Townhouse']

#normality and p values
composite_normality, composite_pvalue = stats.normaltest(composite_prices)
townhouse_normality, townhouse_pvalue = stats.normaltest(townhouse_prices)
one_storey_normality, one_storey_pvalue = stats.normaltest(one_storey_prices)
two_storey_normality, two_storey_pvalue = stats.normaltest(two_storey_prices)
Apartment_normality, Apartment_pvalue = stats.normaltest(apartment_unit_prices)

print("'Composite', Normality: ", composite_normality, " P-value:", composite_pvalue)
print("'Townhouse', Normality: ", townhouse_normality, " P-value:", townhouse_pvalue)
print("'One Storey', Normality: ", one_storey_normality, " P-value:", one_storey_pvalue)
print("'Two Storey', Normality: ", two_storey_normality, " P-value:", two_storey_pvalue)
print("'Apartment', Normality: ", Apartment_normality, " P-value:", Apartment_pvalue)



# Sample plot to ensure you have both python and the data downloaded.
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Composite"],c="red",label="Composite")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["One_storey"],c="blue",label="one-storey")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Two_storey"],c="purple",label="two-storey")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Apartment_unit"],c="yellow",label="Apartment_unit")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Townhouse"],c="green",label="Townhouse")

# creat histogram
plot.figure(figsize=(10, 6))
plot.hist(composite_prices, bins=20, alpha=0.7, color='red', label='Composite Prices')
plot.hist(apartment_unit_prices, bins=20, alpha=0.7, color='yellow', label='Apartment Unit Prices')
plot.hist(one_storey_prices, bins=20, alpha=0.7, color='blue', label='One Storey')
plot.hist(two_storey_prices, bins=20, alpha=0.7, color='purple', label='Two storey')
plot.hist(townhouse_prices, bins=20, alpha=0.7, color='green', label='Townhouse')

plot.title("Comparison of different forms of housing")
plot.ylabel("cost ($)")
plot.xlabel("year of sale")
plot.legend()
plot.show()

print("Hello world! Analysis complete!")

#These data sets are the stats for the listings (may need to be accessed in spark)
#firstQuarterListings = pd.read_csv("Sept_Listing_Data2022.csv")
#secondQuarterListings = pd.read_csv("Dec_Listing_Data2022.csv")
#thirdQuarterListings = pd.read_csv("March_Listing_Data2023.csv")
#fourthQuarterListings = pd.read_csv("June_Listing_Data2023.csv")

## LargeAIRBNBCalendar contains the daily prices of the listings, but doesn't have anything other than the ID associated
## join the listings (Id,Property_type) datasets with the summarised monthly prices (ID, Average price for the month/day)
## We can summarise by month or day if we want. This probably should be done in spark, but feel free to use any method that you know.
