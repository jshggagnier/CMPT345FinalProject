import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sys
from scipy import stats
import pyarrow #reading directory
import scipy.stats as stat
import scipy.optimize as curve
from datetime import datetime

## Clean up our dataset and create new columns
#import datasets and convert columns
AirBNBdata = pd.read_parquet("AirBNB-By-Category")
AirBNBdata["date"] = pd.to_datetime(AirBNBdata["date"])
AirBNBdata['timestamp'] = AirBNBdata['date'].apply(datetime.timestamp)

# Removing outliers
AirBNBdatafiltered = AirBNBdata[(AirBNBdata["timestamp"] != 1695711600.00000)].copy(deep=True)
AirBNBdatafiltered = AirBNBdatafiltered[(AirBNBdatafiltered["timestamp"] != 1717830000.00000)]
AirBNBdatafiltered = AirBNBdatafiltered[(AirBNBdatafiltered["timestamp"] != 1695452400.00000)]
AirBNBdatafiltered = AirBNBdatafiltered[(AirBNBdatafiltered["timestamp"] != 1695366000.00000)]

# Removing odd trend for apartments only
AirBNBdatafilteredAptmnt = AirBNBdatafiltered[(AirBNBdatafiltered["timestamp"] < 1669795200.00000) | (AirBNBdatafiltered["timestamp"] > 1673251200.00000)]

## Creating our linear models
linearStats = stat.linregress(x=AirBNBdatafilteredAptmnt["timestamp"],y = AirBNBdatafilteredAptmnt["Apartment"])
AirBNBdata["Expected Apartment Value"] = AirBNBdata['timestamp']*linearStats.slope + linearStats.intercept
AirBNBdatafilteredAptmnt["Expected Apartment Value"] = AirBNBdatafilteredAptmnt['timestamp']*linearStats.slope + linearStats.intercept
AirBNBdatafilteredAptmnt["apartmentResiduals"] = AirBNBdatafilteredAptmnt["Apartment"] - AirBNBdatafilteredAptmnt["Expected Apartment Value"]

linearStats2 = stat.linregress(x=AirBNBdatafiltered["timestamp"],y = AirBNBdatafiltered["House"])
AirBNBdata["Expected House Value"] = AirBNBdata['timestamp']*linearStats2.slope + linearStats2.intercept
AirBNBdatafiltered["Expected House Value"] = AirBNBdatafiltered['timestamp']*linearStats2.slope + linearStats2.intercept
AirBNBdatafiltered["houseResiduals"] = AirBNBdatafiltered["House"] - AirBNBdatafiltered["Expected House Value"]

## this is the form I expect the function to take
def func(x,a,b,d):
    return a+b*x+14*np.sin(x/d)
    ## 14 is a magic number that I found fits closest to the amplitude,
    ## if I let the model, it will just do a linear model because it doesnt see the maxima

# Fitting the nonlinear curves
popt, pcov = curve.curve_fit(func,xdata=AirBNBdatafiltered["timestamp"],ydata = AirBNBdatafiltered["House"],p0=[linearStats2.intercept,linearStats2.slope,4000000])
AirBNBdatafiltered["nonlinearHouseResiduals"] = AirBNBdatafiltered["House"] - func(AirBNBdatafiltered["timestamp"], *popt)

popt2, pcov2 = curve.curve_fit(func, xdata=AirBNBdatafilteredAptmnt["timestamp"], ydata = AirBNBdatafilteredAptmnt["Apartment"], p0=[linearStats.intercept, linearStats.slope,4000000])
AirBNBdatafilteredAptmnt["nonlinearApartmentResiduals"] = AirBNBdatafilteredAptmnt["Apartment"] - func(AirBNBdatafilteredAptmnt["timestamp"], *popt2)

## Statistical analysis
# correlation coefficient
print(AirBNBdata[["Apartment","House"]].corr())

#normality tests of residuals of the models
normality, p_value = stats.normaltest(AirBNBdatafiltered["houseResiduals"])
print(f"House, Normality: {normality}, P-value: {p_value}")

normality, p_value = stats.normaltest(AirBNBdatafilteredAptmnt["apartmentResiduals"])
print(f"Apartment, Normality: {normality}, P-value: {p_value}")

normality, p_value = stats.normaltest(AirBNBdatafiltered["nonlinearHouseResiduals"])
print(f"House, Nonlinear Normality: {normality}, P-value: {p_value}")

normality, p_value = stats.normaltest(AirBNBdatafilteredAptmnt["nonlinearApartmentResiduals"])
print(f"Apartment, Nonlinear Normality: {normality}, P-value: {p_value}")

# Plotting the models
plot.figure(figsize=(9, 6))
plot.scatter(AirBNBdata["date"], AirBNBdata["Apartment"],marker="x", c="Red", label="Removed for Training regression")
plot.scatter(AirBNBdata["date"], AirBNBdata["House"],marker="x", c="Green")
plot.scatter(AirBNBdatafilteredAptmnt["date"], AirBNBdatafilteredAptmnt["Apartment"], c="Red", label="Apartment")
plot.scatter(AirBNBdatafiltered["date"], AirBNBdatafiltered["House"], c="Green", label="House")
plot.plot(AirBNBdata["date"], AirBNBdata["Expected House Value"],ls="-",linewidth="5",c="DarkRed", label="FittedHouse")
plot.plot(AirBNBdata["date"], AirBNBdata["Expected Apartment Value"],ls="-",linewidth="5",c="DarkGreen", label="FittedApartment")
plot.plot(AirBNBdata["date"], func(AirBNBdata["timestamp"], *popt), ls='-',c="Blue",linewidth=3,label='nonlinear Fitted House')
plot.plot(AirBNBdata["date"], func(AirBNBdata["timestamp"], *popt2), ls='-',c="Blue",linewidth=3,label='nonlinear Fitted Apartment')
plot.title("AirBNB Data OLS Regression")
plot.ylabel("Prices ($)")
plot.xlabel("Year of Rental Taking Place")
plot.legend()
plot.show()

plot.figure(figsize=(9, 6),dpi=900)
plot.subplot(221)
plot.suptitle("AirBNB Data Residual Histograms")
plot.hist(AirBNBdatafiltered["houseResiduals"])
plot.title("Housing")
plot.ylabel("Count")

plot.subplot(222)
plot.hist(AirBNBdatafilteredAptmnt["apartmentResiduals"])
plot.title("Apartments")
plot.ylabel("Count")

plot.subplot(223)
plot.hist(AirBNBdatafiltered["nonlinearHouseResiduals"])
plot.title("Non-Linear")
plot.ylabel("Count")
plot.xlabel("Residual ($)")

plot.subplot(224)
plot.hist(AirBNBdatafilteredAptmnt["nonlinearApartmentResiduals"])
plot.title("Non-Linear")
plot.ylabel("Count")
plot.xlabel("Residual ($)")
plot.savefig("histograms.png")