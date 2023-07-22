import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

## data frame contains 221 months worth of average prices for houses of varying types, first column is date
monthlyAvgPrice = pd.read_csv("ByApartment.csv", parse_dates=['Date'])

plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Apartment_unit"],c="blue",label="apartments")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Two_storey"],c="red",label="two-storey")
plot.title("comparison of different forms of housing")
plot.ylabel("cost ($)")
plot.xlabel("year of sale")
plot.legend()
plot.show()

print("Hello world! Analysis complete!")