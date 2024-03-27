import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("FRED-NROUST.csv")
df.head()

df.plot()
plt.show()


######### Paritioning the data ##############
y = df['Value']
y_train = df['Value'][:-8]
y_test = df['Value'][-8:]


plt.plot(y_train, color='blue', label='Train')
plt.plot(y_test, color='orange', label='Test')
plt.legend(loc='best')
plt.show()


#### Centered MA
fcast = y.rolling(3,center=True).mean()
plt.plot(y, label='Original Data')
plt.plot(fcast, label='Centered MA')
plt.legend(loc='best')
plt.show()

span=6
#### Trailing MA
fcast = y_train.rolling(span).mean()
last_val = fcast.iloc[-1]
MA_series = pd.Series(last_val.repeat(len(y_test)))
MA_fcast = pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(MA_fcast, label='Moving Average')
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, MA_series))
print("RMSE =",rmse)

################### SES ##################
from statsmodels.tsa.api import SimpleExpSmoothing

# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(y_train).fit()
fcast1 = fit1.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast1, label='SES')
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast1))
print("RMSE =",rmse)


############### Holt's Linear ##################
from statsmodels.tsa.api import Holt
alpha=0.1
beta=0.9
fit2 = Holt(y_train).fit()
fcast2 = fit2.forecast(len(y_test))

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast2, label="Holt's Linear")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast2))
print("RMSE =",rmse)

############### Holt's Exponential ##################
from statsmodels.tsa.api import Holt

fit2 = Holt(y_train,exponential=True).fit()
fcast2 = fit2.forecast(len(y_test))

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast2, label="Holt's Exponential")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast2))
print("RMSE =",rmse)

############### Additive Trend ##################
from statsmodels.tsa.api import Holt
alpha=0.1
beta=0.8
phi=0.2
fit2 = Holt(y_train,damped_trend=True).fit()
fcast2 = fit2.forecast(len(y_test))

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast2, label="Additive Damped Trend")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast2))
print("RMSE =",rmse)

############### Multiplicative Trend ##################
from statsmodels.tsa.api import Holt

fit2 = Holt(y_train,damped_trend=True,exponential=True).fit()
fcast2 = fit2.forecast(len(y_test))

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast2, label="Multiplicative Damped Trend")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast2))
print("RMSE =",rmse)

################## Holt-Winters Additive ####################
from statsmodels.tsa.api import ExponentialSmoothing

fit3 = ExponentialSmoothing(y_train,seasonal_periods=4, 
                            trend='add', seasonal='add').fit()
fcast3 = fit3.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast3, label="Holt-Winters Additive")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast3))
print("RMSE =",rmse)


################## Holt-Winters Multiplicative ####################
from statsmodels.tsa.api import ExponentialSmoothing

fit3 = ExponentialSmoothing(y_train,seasonal_periods=4, 
                            trend='add', seasonal='mul').fit()
fcast3 = fit3.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast3, label="Holt-Winters Multiplicative")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast3))
print("RMSE =",rmse)



################## Damped Holt-Winters Additive ####################
from statsmodels.tsa.api import ExponentialSmoothing

fit3 = ExponentialSmoothing(y_train,seasonal_periods=4, damped_trend=True,
                            trend='add', seasonal='add').fit()
fcast3 = fit3.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast3, label="Damped Holt-Winters Additive")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast3))
print("RMSE =",rmse)


################## Damped Holt-Winters Multiplicative ####################
from statsmodels.tsa.api import ExponentialSmoothing

fit3 = ExponentialSmoothing(y_train,seasonal_periods=4, damped_trend=True, 
                            trend='add', seasonal='mul').fit()
fcast3 = fit3.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast3, label="Damped Holt-Winters Multiplicative")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast3))
print("RMSE =",rmse)

####################### BUNDESBANK-BBK01_WT5511.csv ###################
df = pd.read_csv("BUNDESBANK-BBK01_WT5511.csv")


######### Paritioning the data ##############
y = df['Value']
y_train = df['Value'][:-12]
y_test = df['Value'][-12:]


plt.plot(y_train, color='blue', label='Train')
plt.plot(y_test, color='orange', label='Test')
plt.legend(loc='best')
plt.show()


#### Centered MA
fcast = y.rolling(3,center=True).mean()
plt.plot(y, label='Original Data')
plt.plot(fcast, label='Centered MA')
plt.legend(loc='best')
plt.show()

span=5
#### Trailing MA
fcast = y_train.rolling(span).mean()
last_val = fcast.iloc[-1]
MA_series = pd.Series(last_val.repeat(len(y_test)))
MA_fcast = pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(MA_fcast, label='Moving Average')
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, MA_series))
print("RMSE =",rmse)

################### SES ##################
from statsmodels.tsa.api import SimpleExpSmoothing

# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(y_train).fit()
fcast1 = fit1.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast1, label='SES')
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast1))
print("RMSE =",rmse)


############### Holt's Linear ##################
from statsmodels.tsa.api import Holt
alpha=0.1
beta=0.9
fit2 = Holt(y_train).fit()
fcast2 = fit2.forecast(len(y_test))

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast2, label="Holt's Linear")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast2))
print("RMSE =",rmse)

############### Holt's Exponential ##################
from statsmodels.tsa.api import Holt

fit2 = Holt(y_train,exponential=True).fit()
fcast2 = fit2.forecast(len(y_test))

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast2, label="Holt's Exponential")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast2))
print("RMSE =",rmse)

############### Additive Damped Trend ##################
from statsmodels.tsa.api import Holt
alpha=0.1
beta=0.8
phi=0.2
fit2 = Holt(y_train,damped_trend=True).fit()
fcast2 = fit2.forecast(len(y_test))

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast2, label="Additive Damped Trend")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast2))
print("RMSE =",rmse)

############### Damped Multiplicative Trend ##################
from statsmodels.tsa.api import Holt

fit2 = Holt(y_train,damped_trend=True,exponential=True).fit()
fcast2 = fit2.forecast(len(y_test))

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast2, label="Multiplicative Damped Trend")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast2))
print("RMSE =",rmse)

################## Holt-Winters Additive ####################
from statsmodels.tsa.api import ExponentialSmoothing

fit3 = ExponentialSmoothing(y_train,seasonal_periods=12, 
                            trend='add', seasonal='add').fit()
fcast3 = fit3.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast3, label="Holt-Winters Additive")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast3))
print("RMSE =",rmse)


################## Holt-Winters Multiplicative ####################
from statsmodels.tsa.api import ExponentialSmoothing

fit3 = ExponentialSmoothing(y_train,seasonal_periods=12, 
                            trend='add', seasonal='mul').fit()
fcast3 = fit3.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast3, label="Holt-Winters Multiplicative")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast3))
print("RMSE =",rmse)



################## Damped Holt-Winters Additive ####################
from statsmodels.tsa.api import ExponentialSmoothing

fit3 = ExponentialSmoothing(y_train,seasonal_periods=12, damped_trend=True,
                            trend='add', seasonal='add').fit()
fcast3 = fit3.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast3, label="Damped Holt-Winters Additive")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast3))
print("RMSE =",rmse)


################## Damped Holt-Winters Multiplicative ####################
from statsmodels.tsa.api import ExponentialSmoothing

fit3 = ExponentialSmoothing(y_train,seasonal_periods=12, damped_trend=True, 
                            trend='add', seasonal='mul').fit()
fcast3 = fit3.forecast(len(y_test))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(fcast3, label="Damped Holt-Winters Multiplicative")
plt.legend(loc='best')
plt.show()

#### Evaluating
rmse = np.sqrt(mean_squared_error(y_test, fcast3))
print("RMSE =",rmse)
