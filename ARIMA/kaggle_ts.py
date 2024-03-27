import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir(r"C:\Training\Kaggle\Datasets\House Property Sales Time Series")
df = pd.read_csv("raw_sales.csv", parse_dates=['datesold'])
df['year'] = df['datesold'].dt.year
df['month'] = df['datesold'].dt.month

total_sales = df.groupby(['year','month'])['price'].sum()
total_sales = total_sales.reset_index()

y = total_sales['price']
y_train = y[:-6]
y_test = y[-6:]

plt.plot(y_train, color='blue', label='Train')
plt.plot(y_test, color='orange', label='Test')
plt.legend(loc='best')
plt.show()


