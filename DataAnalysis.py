import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sys
from scipy import stats
import pyarrow #reading directory 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

# Equality of variances
levene_test, levene_pvalue = stats.levene(composite_prices, apartment_unit_prices, one_storey_prices, two_storey_prices, townhouse_prices)
print("Levene Test - Equality of Variances:")
print("Statistic:", levene_test)
print("P-value:", levene_pvalue)

# Correlation Coefficient
correlation_coefficients = monthlyAvgPrice[['Composite', 'Apartment_unit', 'One_storey', 'Two_storey', 'Townhouse']].corr()
print ("Correlation Coefficients is: ","\n",correlation_coefficients)

# Sample plot to ensure you have both python and the data downloaded.
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Composite"],c="red",label="Composite")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["One_storey"],c="blue",label="one-storey")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Two_storey"],c="purple",label="two-storey")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Apartment_unit"],c="yellow",label="Apartment_unit")
plot.scatter(monthlyAvgPrice["Date"],monthlyAvgPrice["Townhouse"],c="green",label="Townhouse")
plot.title("Comparison of different forms of housing")
plot.ylabel("Prices ($)")
plot.xlabel("Year of Sale")
plot.legend()
plot.show()

# Histogram
plot.figure(figsize=(10, 6))
plot.hist(composite_prices, bins=20, alpha=0.7, color='red', label='Composite Prices')
plot.hist(apartment_unit_prices, bins=20, alpha=0.7, color='yellow', label='Apartment Unit Prices')
plot.hist(one_storey_prices, bins=20, alpha=0.7, color='blue', label='One Storey')
plot.hist(two_storey_prices, bins=20, alpha=0.7, color='purple', label='Two storey')
plot.hist(townhouse_prices, bins=20, alpha=0.7, color='green', label='Townhouse')

plot.title("Comparison of different forms of housing")
plot.ylabel("Prices ($)")
plot.xlabel("year of sale")
plot.legend()
plot.show()

##  END OF HOUSING 


## start of AIRBNB Analysis
AirBNBApartment = AirBNBdata['Apartment']
AirBNBHouse = AirBNBdata['House']
AirBNBDates = AirBNBdata['date']

# normality and p values

apartment_normality, apartment_pvalue = stats.normaltest(AirBNBApartment)
house_normality, house_pvalue = stats.normaltest(AirBNBHouse)

print("'AirBNB Apartment', Normality: ", apartment_normality, " P-value:", apartment_pvalue)
print("'AirBNB House', Normality: ", house_normality, " P-value:", house_pvalue)


# Equality of variances
levene_test, levene_pvalue = stats.levene(AirBNBApartment, AirBNBHouse)
print("Levene Test - Equality of Variances:")
print("Statistic:", levene_test)
print("P-value:", levene_pvalue)

plot.scatter(AirBNBdata["date"], AirBNBdata["Apartment"], c="red", label="Apartment")
plot.scatter(AirBNBdata["date"], AirBNBdata["House"], c="blue", label="House")
plot.title("AirBNB Data Solo")
plot.ylabel("Prices ($)")
plot.xlabel("year of rental taking place")
plot.legend()
plot.show()

## creating comparison ranges
MonthlyAirBNB = AirBNBdata.groupby(pd.Grouper(key='date', freq='1MS')).mean()
print(MonthlyAirBNB.index.tolist())
plot.scatter(MonthlyAirBNB.index.tolist(), MonthlyAirBNB["Apartment"], c="red", label="Apartment")
plot.scatter(MonthlyAirBNB.index.tolist(), MonthlyAirBNB["House"], c="blue", label="House")
plot.title("AirBNB data Monthly Solo")
plot.ylabel("Prices ($)")
plot.xlabel("Year of rental taking place")
plot.legend()
plot.show()

print("Analysis complete!")

## END of AirBNB

# Forecasting 
# Linear regression
# Predict x property type pricing with a given property type  pricing

def predict_property_prices(predictor, target, df, future_periods, correlation_coefficients):
    
    # Columns 
    predictor_prices = df[predictor]
    target_prices = df[target]
    
    # Linear regression
    X = predictor_prices.values.reshape(-1, 1)
    y = target_prices.values.reshape(-1, 1)

    # training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) #random_state is for testing

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make prediction
    future_predictor_prices = np.linspace(min(predictor_prices), max(predictor_prices), future_periods).reshape(-1, 1)
    future_target_prices = model.predict(future_predictor_prices)

    # Plot original and predicted 
    plot.scatter(predictor_prices, target_prices, c="red", label="Actual Prices")
    plot.plot(future_predictor_prices, future_target_prices, c="blue", label="Predicted Prices")
    #using f to make sure it doesnt print {target_property}
    plot.xlabel(f"{predictor} Prices ($)")
    plot.ylabel(f"{target} Prices ($)")
    plot.title(f"Predicting {target} Prices using {predictor} Prices")
    plot.legend()
    plot.show()

    # Find the best predictor for apartment example
    correlation = correlation_coefficients.loc['Apartment_unit']
    best_predictor = correlation.drop('Apartment_unit').idxmax()
    print("Best Predictor for Apartment Prices:", best_predictor)
    
# example: using apartment prices to make a prediction 

predict_property_prices ('Apartment_unit', 'Composite', monthlyAvgPrice, 24,correlation_coefficients)
predict_property_prices ('Apartment_unit', 'One_storey', monthlyAvgPrice, 24,correlation_coefficients)
predict_property_prices ('Apartment_unit', 'Two_storey', monthlyAvgPrice, 24,correlation_coefficients)
predict_property_prices ('Apartment_unit', 'Townhouse', monthlyAvgPrice, 24,correlation_coefficients)

## END of forecastin

## END of analysis 