import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sys
from scipy import stats
import pyarrow #reading directory 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Getting Data
## Data frame contains 221 months worth of average prices for houses of varying types, first column is date
monthlyAvgPrice = pd.read_csv("ByApartment.csv", parse_dates=['Date'])

## Data frame contains 221 months worth of average prices for houses of varying types, first column is date
monthlyAvgPrice = pd.read_csv("ByApartment.csv", parse_dates=['Date'])

## Read spark output into pandas!
AirBNBdata = pd.read_parquet("AirBNB-By-Category")
AirBNBdata["date"] = pd.to_datetime(AirBNBdata["date"])

# column data
columns = ['Composite', 'Apartment_unit', 'One_storey', 'Two_storey', 'Townhouse']
labels = ['Composite Prices', 'Apartment Unit Prices', 'One Storey Prices', 'Two Storey Prices', 'Townhouse Prices']
data = [monthlyAvgPrice[col] for col in columns]

# normality and p value
def print_normality_and_p_value(data, label):
    normality, p_value = stats.normaltest(data)
    print(f"'{label}', Normality: {normality}, P-value: {p_value}")
    
# Print normality and p values
print("\n, Part 1 - Housing Analysis")
for col, label in zip(columns, labels):
    print_normality_and_p_value(monthlyAvgPrice[col], label)

levene_test, levene_pvalue = stats.levene(*data)
print("Levene Test - Equality of Variances:")
print("Statistic:", levene_test)
print("P-value:", levene_pvalue)

# Correlation Coefficient
correlation_coefficients = monthlyAvgPrice[columns].corr()
print("Correlation Coefficients is: \n", correlation_coefficients)

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


print(" -- End of Housing Analysis complete -- \n")
##  END OF HOUSING 


## Start of AIRBNB Analysis
# column list
Airbnb_columns = ['Apartment', 'House']
Airbnb_labels = ['Apartment Prices', 'House Price']
Airbnb_data = [AirBNBdata[col] for col in Airbnb_columns]

# Print normality and p values
print("\n", "Part 2 - AIRBNB Analysis")
for col, label in zip(Airbnb_columns, Airbnb_labels):
    print_normality_and_p_value(AirBNBdata[col], label)

levene_test, levene_pvalue = stats.levene(*Airbnb_data)
print("Levene Test - Equality of Variances:")
print("Statistic:", levene_test)
print("P-value:", levene_pvalue)

plot.scatter(AirBNBdata["date"], AirBNBdata["Apartment"], c="red", label="Apartment")
plot.scatter(AirBNBdata["date"], AirBNBdata["House"], c="blue", label="House")
plot.title("AirBNB Data Solo")
plot.ylabel("Prices ($)")
plot.xlabel("Year of Rental Taking Place")
plot.legend()
plot.show()

# Correlation Coefficient
correlation_coefficients_AirBNB = AirBNBdata[['Apartment', 'House']].corr()
print ("Correlation Coefficients is: ","\n",correlation_coefficients_AirBNB)

## creating comparison ranges
MonthlyAirBNB = AirBNBdata.groupby(pd.Grouper(key='date', freq='1MS')).mean()
#print(MonthlyAirBNB.index.tolist())
plot.scatter(MonthlyAirBNB.index.tolist(), MonthlyAirBNB["Apartment"], c="red", label="Apartment")
plot.scatter(MonthlyAirBNB.index.tolist(), MonthlyAirBNB["House"], c="blue", label="House")
plot.title("AirBNB data Monthly Solo")
plot.ylabel("Prices ($)")
plot.xlabel("Year of rental taking place")
plot.legend()
plot.show()

print(" -- End of Airbnb Analysis complete -- \n ")

## END of AirBNB

# Forecasting 
# Linear regression
# Predict x property type pricing with a given property type  pricing
print ('\n',"Forecasting Analysis")
def predict_property_prices(predictor, target, df, future_periods, correlation_coefficients):
    
    # Columns 
    predictor_prices = df[predictor]
    target_prices = df[target]
    
    # Linear Regression Model
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

    # Find the best predictor for the apartment example
    correlation = correlation_coefficients.loc['Apartment_unit']
    best_predictor = correlation.drop('Apartment_unit').idxmax()
    print("Best Predictor for Apartment Prices:", best_predictor)
    
# Example: using apartment prices to make a prediction 

predict_property_prices ('Apartment_unit', 'Composite', monthlyAvgPrice, 24,correlation_coefficients)
predict_property_prices ('Apartment_unit', 'One_storey', monthlyAvgPrice, 24,correlation_coefficients)
predict_property_prices ('Apartment_unit', 'Two_storey', monthlyAvgPrice, 24,correlation_coefficients)
predict_property_prices ('Apartment_unit', 'Townhouse', monthlyAvgPrice, 24,correlation_coefficients)

## END of Forecasting

## END of Analysis 