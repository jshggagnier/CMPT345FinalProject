import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sys
from scipy import stats


## data frame contains 221 months worth of average prices for houses of varying types, first column is date
monthlyAvgPrice = pd.read_csv("ByApartment.csv", parse_dates=['Date'])

## This is the easiest way I can think of to read spark output into pandas!

AirBNBdata = pd.read_parquet("AirBNB-By-Category")
AirBNBdata["date"] = pd.to_datetime(AirBNBdata["date"])

# data column
composite_prices = monthlyAvgPrice['Composite']
apartment_unit_prices = monthlyAvgPrice['Apartment_unit']
one_storey_prices = monthlyAvgPrice['One_storey']
two_storey_prices = monthlyAvgPrice['Two_storey']
townhouse_prices = monthlyAvgPrice ['Townhouse']

# normality and p values
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

# Levene test for equality of variances
levene_test, levene_pvalue = stats.levene(composite_prices, apartment_unit_prices, one_storey_prices, two_storey_prices, townhouse_prices)
print("Levene Test - Equality of Variances:")
print("Statistic:", levene_test)
print("P-value:", levene_pvalue)

#correlation coefficient
correlation_coefficients = monthlyAvgPrice[['Composite', 'Apartment_unit', 'One_storey', 'Two_storey', 'Townhouse']].corr()
print ("Correlation Coefficients is: ","\n",correlation_coefficients)

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

## start of AIRBNB Analysis
plot.scatter(AirBNBdata["date"], AirBNBdata["Apartment"], c="red", label="Apartment")
plot.scatter(AirBNBdata["date"], AirBNBdata["House"], c="blue", label="House")
plot.title("AirBNB Data Solo")
plot.ylabel("cost ($)")
plot.xlabel("year of rental taking place")
plot.legend()
plot.show()

## creating comparison ranges
MonthlyAirBNB = AirBNBdata.groupby(pd.Grouper(key='date', freq='1MS')).mean()
print(MonthlyAirBNB.index.tolist())
plot.scatter(MonthlyAirBNB.index.tolist(), MonthlyAirBNB["Apartment"], c="red", label="Apartment")
plot.scatter(MonthlyAirBNB.index.tolist(), MonthlyAirBNB["House"], c="blue", label="House")
plot.title("AirBNB data Monthly Solo")
plot.ylabel("cost ($)")
plot.xlabel("year of rental taking place")
plot.legend()
plot.show()

# forecasting 
