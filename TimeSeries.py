import pandas as pd
import matplotlib.pyplot as mp
from pmdarima.arima import auto_arima 

#Read Data from csv
data=pd.read_csv("Power-Networks-LCL.csv") 

#Convert the Data type of DateTime Column to DateTime
data['DateTime'] = pd.to_datetime(data['DateTime'],infer_datetime_format=True)

#Find Top3 household that has more samples
top3household = data.LCLid.value_counts().to_frame().head(3)
print("Top 3 HouseHolds with highest number of samples")
print("-----------------------------------------------")
print(top3household)

#Find the Start and End Date to split the Data Further in the code
print("\nStart Date in the DataFile", data['DateTime'].min())
print("Start Date in the DataFile",data['DateTime'].max())

for index, row in top3household.iterrows():
    #print(index, row['LCLid'])
    #Fetch the individual household data
    print("\nFeature Prediction for HouseHold : ",index)
    print("---------------------------------------------")
    dfperhousehold = data[data['LCLid'] == index]
    
    # Re-sampling the half hour data to hourly Data
    hourdata=dfperhousehold.resample('60min',on='DateTime').sum()

    # Re-sampling the half hour data to Daily Data
    # We are Going to Use this further in the prediction as Daily data resulting in memory error
    dailydata=dfperhousehold.resample('D',on='DateTime').sum()
    
    #print(dailydata)    
    
    #Build the Auto-Arima model for the Daily Data and Print the Values
    buildmodel = auto_arima(dailydata, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    
    print("\nAuto Arima AIC Value : ",buildmodel.aic())
    
    #Split Dataframe to Train and test dataset
    train_data=dailydata['2011-12-06':'2013-09-30'] 
    valid_data=dailydata['2013-10-01':'2014-02-28']
    valid_data=valid_data.rename(columns={'KWh':'Test KWH Values'})

    buildmodel.fit(train_data)
    forecast_value = buildmodel.predict(n_periods=len(valid_data))
    forecast_value = pd.DataFrame(forecast_value,index = valid_data.index,columns=['Predicted KWH Values'])
    print("\nFeature Forecast Value : \n",forecast_value)
    
    
    pd.concat([valid_data,forecast_value],axis=1).plot(figsize=(10,8))
    mp.title('Compare Test Data With Predicted Data for HouseHold %s' %index, fontsize=14)
    mp.xlabel("Datetime") 
    mp.ylabel("KWH") 
    mp.legend(loc='best') 
    mp.show()

    pd.concat([dailydata,forecast_value],axis=1).plot(figsize=(10,8))
    mp.title( 'Compare Prediction data With Entire Data for HouseHold %s' %index, fontsize=14)
    mp.xlabel("Datetime") 
    mp.ylabel("KWH") 
    mp.legend(loc='best') 
    mp.show()