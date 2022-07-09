#Submitted by : B20191 Deepak Kumar

# importing required modules:
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg as AR

df_q1=pd.read_csv("daily_covid_cases.csv")
original=df_q1['new_cases']

print("-----------------------Question_1_part_a-------------------")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(df_q1['Date'],
           df_q1['new_cases'].values,
           color='purple')
ax.set(xlabel="Date", ylabel="new_cases",
       title="Q1 part a")
date_form = DateFormatter("%b-%d")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation =45)
plt.show()

print("-----------------------Question_1_part_b-------------------")
one_day_lag=original.shift(1)
print("Q1 part b --> Pearson correlation (autocorrelation) coefficient :",original.corr(one_day_lag))
print()

print("-----------------------Question_1_part_c-------------------")
plt.scatter(original, one_day_lag, s=5)
plt.xlabel("Given time series data")
plt.ylabel("One day lagged time series data")
plt.title("Q1 part c")
plt.show()
print("-----------------------Question_1_part_d-------------------")
PCC=sm.tsa.acf(original)
lag=[1,2,3,4,5,6]
pcc=PCC[1:7]
plt.plot(lag,pcc, marker='o')
for xitem,yitem in np.nditer([lag, pcc]):
        etiqueta = "{:.3f}".format(yitem)
        plt.annotate(etiqueta, (xitem,yitem), textcoords="offset points",xytext=(0,10),ha="center")
plt.xlabel("Lag value")
plt.ylabel("Correlation coffecient value")
plt.title("Q1 part d")
plt.show()

print("-----------------------Question_1_part_e-------------------")
plot_acf(x=original, lags=50)
plt.xlabel("Lag value")
plt.ylabel("Correlation coffecient value")
plt.title("Q1 part e")
plt.show()

#Q-2
print("-----------------------Question_2-------------------")
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
train_size = 0.65 # 35% for testing
X = series.values
train, test = X[:int(len(X)*train_size)], X[int(len(X)*train_size):]

window = 5 # The lag=1
model = AR(train, lags=window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print()
print("Q2 part a--> coefficients are :",coef)
print()
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

### Q2 part b,,, part 1
plt.scatter(test,predictions )
plt.xlabel('Actual cases')
plt.ylabel('Predicted cases')
plt.title('Q2 part b\n Part 1')
plt.show()

### Q2 part b,,, part 2
x=[i for i in range(len(test))]
plt.plot(x,test, label='Actual cases')
plt.plot(x,predictions , label='Predicted cases')
plt.legend()
plt.title('Q2 part b\n Part 2')
plt.show()

### Q2 part b,,, part 3
rmse=mean_squared_error(test, predictions,squared=False)
print("Q2 part b-1--> persent RMSE :",rmse*100/(sum(test)/len(test)),"%")
print()

mape=mean_absolute_percentage_error(test, predictions)
print("Q2 part b-1--> persent MAPE :",mape)

#Q-3
print("-----------------------Question_3-------------------")
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
train_size = 0.65 # 35% for testing
X = series.values
train, test = X[:int(len(X)*train_size)], X[int(len(X)*train_size):]

def ARmodel(train_data, test_data, lag):
    window=lag
    model = AR(train_data, lags=window)
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model

    #using these coefficients walk forward over time steps in test, one step each time
    history = train_data[len(train_data)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test_data)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1] # Add other values
        obs = test_data[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.'''
    rmse_=mean_squared_error(test_data, predictions,squared=False)*100/(sum(test_data)/len(test_data))
    mape_=mean_absolute_percentage_error(test_data, predictions)
    return rmse_, mape_

lag=[1,5,10,15,25]
rmse_list=[]
mape_list=[]
for i in lag:
    rmse, mape=ARmodel(train, test,i)
    rmse_list.append(rmse[0])
    mape_list.append(mape)

plt.bar(lag, rmse_list)
plt.ylabel('RMSE error')
plt.xlabel('Lag values')
plt.title("Q3\n Bar chart between RMSE and Lag values")
plt.xticks(lag)
plt.show()

plt.bar(lag, mape_list)
plt.ylabel('MAPE error')
plt.xlabel('Lag values')
plt.title("Q3\n Bar chart between MAPE and Lag values")
plt.xticks(lag)
plt.show()

print("-----------------------Question_4-------------------")
df_q3=pd.read_csv("daily_covid_cases.csv")
train_q4=df_q3.iloc[:int(len(df_q3)*0.65)]
train_q4=train_q4['new_cases']
i=0
corr = 1
# abs(AutoCorrelation) > 2/sqrt(T)
while corr > 2/(len(train_q4))**0.5:
    i += 1
    t_new = train_q4.shift(i)
    corr = train_q4.corr(t_new)
print(i)
rmse_q4, mape_q4=ARmodel(train, test, i)
print("Q4--> Lag(heuristic) value is :", i)
print(f"Q4--> RMSE value for lag value = {i} is :",rmse_q4[0])
print(f"Q4--> MAPE value for lag value = {i} is :",mape_q4)