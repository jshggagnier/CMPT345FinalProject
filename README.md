# CMPT353FinalProject
Our data analysis project, with all data and python code




Requirement: 
    - Local Pyspark
    - Libraries: Pandas, Matplotlib, Numpy, Pyarrow, Sklearn, Scipy.stat, Scipy.optimize 


All code does not require commandline prompts, as it has hardcoded directories. Simply git pull and run the files using python (with all requirements pip installed).


Part 1 - Data Cleaning (Run using local Pyspark) 
(not necessary as all results have been uploaded) 

    - data cleaning: filtering data and cleaning values is done in spark 
        - data is also further categorized into small and large data sets
            - Small datasets shows the listings
				- Cleaned using AirbnbListingCleaning.py
				- categorized listings into 2 main categories, Apartment rentals, and House rentals
            - Large shows the bookings in each listing 
				-Performed in AirbnbBookingCleaning.py
				-for example, Listings that did not get booked were filtered out as they are not relevant
        - Listing and Bookings "PysparkDataMerge" are then joined together after for analysis 

Part 2 - Data Analysis 
Analyzed the correlation between different types of buildings for both separate datasets

    a) Housing (DataAnalysis.py) 
    Analyze on the price correlation between different property types (apartment, townhouse, one storey, two storey, and composite)
        - Fit a linear model as a predictor, and determine if the residuals are normally distributed
        - Plot diagram to demostrate 
        - Use a forecasting model to predict one type on property prices using one type of property price
        - Determine which one is the best predictor using property type x
        
    b) Airbnb (AirBNBDataAnalysis.py)
		- Fit a linear model as a predictor, determine if residuals are normally distributed
		- analyse outliers, and determine if they will improve the model if removed
		- fit a nonlinear model to better predict the data, as there is a nonlinear trend
		- plot the residuals before and after
	
Easy pip access:

pip install pyarrow

pip install numpy

pip install pandas

pip install matplotlib

pip install sklearn

pip install scipy


Note: any commits done by the following people are actually:

Littlebigone = Daphne

pklucky = Parham

Jshggagnier = Joshua
