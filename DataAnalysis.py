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

def to_timestamp(x):
    return x.timestamp()

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

monthlyAvgPrice['timestamp'] = monthlyAvgPrice['Date'].apply(to_timestamp)
def getResiduals(col):
    fit = stats.linregress(monthlyAvgPrice['timestamp'], monthlyAvgPrice[col])
    prediction_df = pd.DataFrame()
    prediction_df['timestamp'] = monthlyAvgPrice['timestamp']
    prediction_df['prediction'] = monthlyAvgPrice['timestamp']*fit.slope + fit.intercept
    residuals = monthlyAvgPrice[col] - prediction_df['prediction']
    normality, p_value = stats.normaltest(residuals)
    plot.xticks(rotation=25)
    plot.plot(monthlyAvgPrice['Date'], monthlyAvgPrice[col], 'b.', alpha=0.5, label='Actual Data')
    plot.plot(monthlyAvgPrice['Date'], prediction_df['prediction'], 'r-', linewidth=3, label='Best Fitted Line')
    plot.title(f'Price Over Time - {col}')
    plot.xlabel('Date')
    plot.ylabel('Price')
    plot.legend()
    plot.show()

    plot.hist(residuals, bins=15)
    plot.title(f'Histogram of {col} Residuals')
    plot.show()
    print(f"P Value of {col} Residuals:", p_value)

for col in columns:
    getResiduals(col)

print(" -- End of Housing Analysis complete -- \n")
##  END OF HOUSING 


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