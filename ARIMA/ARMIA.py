import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

df = pd.read_csv("AusGas.csv")
df.head()


df.plot.line(x = 'Month',y = 'GasProd')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['GasProd'], lags=20)
plt.show()

y = df['GasProd']
y_train = y[:464]
y_test = y[464:]


#################### AR ############################
from statsmodels.tsa.ar_model import AutoReg
# train autoregression
model = AutoReg(y_train, lags=3)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)

error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))
# plot results
plt.plot(y_test, label='Test')
plt.plot(predictions, color='red', label='Predicted')
plt.legend(loc='best')
plt.show()


# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
predictions.plot(color="purple", label='Forcast')
plt.legend(loc='best')
plt.show()

########################## MA ##############################
from statsmodels.tsa.arima.model import ARIMA
# train MA
model = ARIMA(y_train,order=(0,0,2))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))

# plot results
plt.plot(y_test, label='Test')
plt.plot(predictions, color='red', label='Predictions')
plt.legend(loc='best')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")
plt.show()


########################## ARMA ##############################
from statsmodels.tsa.arima.model import ARIMA
# train MA
model = ARIMA(y_train,order=(2,0,2))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))

# plot results
plt.plot(y_test, label='Test')
plt.plot(predictions, color='red', label='Predictions')
plt.legend(loc='best')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")
plt.show()

###################### Dicky Fuller Test ######################
from statsmodels.tsa.stattools import adfuller

def adfuller_test(ts):
    adfuller_result = adfuller(ts, autolag=None)
    adfuller_out = pd.Series(adfuller_result[0:4],
    index=[ 'Test Statistic',
    'p-value',
    'Lags Used',
    'Number of Observations Used'])
    print(adfuller_out)

adfuller_test(y_train)

########################## ARIMA ##############################
from statsmodels.tsa.arima.model import ARIMA
# train MA
model = ARIMA(y_train,order=(4,1,2))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))

# plot results
plt.plot(y_test, label='Test')
plt.plot(predictions, color='red', label='Predictions')
plt.legend(loc='best')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")
plt.show()

################### Auto ARIMA ############################
from pmdarima.arima import auto_arima

model = auto_arima(y_train, trace=True,
                   error_action='ignore', 
                   suppress_warnings=True)

forecast = model.predict(n_periods=len(y_test))

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Valid',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
plt.show()

# plot results
plt.plot(y_test, label='Test')
plt.plot(forecast, color='red', label='Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test RMSE: %.3f' % rms)

########################### SARIMA #############################

model = auto_arima(y_train, trace=True,error_action='ignore', 
                   suppress_warnings=True, seasonal=True,m=12)

forecast = model.predict(n_periods=len(y_test))

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Valid',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
plt.show()

# plot results
plt.plot(y_test, label='Test')
plt.plot(forecast, color='red', label='Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test RMSE: %.3f' % rms)





