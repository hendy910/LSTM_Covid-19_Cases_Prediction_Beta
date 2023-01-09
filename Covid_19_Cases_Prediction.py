# %% 
# Import necessary packages
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from keras.utils import plot_model

# %%
# 1. Data Loading
CSV_PATH_TRAIN = os.path.join(os.getcwd(), 'Train', 'cases_malaysia_train.csv')
train_df = pd.read_csv(CSV_PATH_TRAIN)

# %%
# 2. Data Inspection
train_df.info() # Open is in Object, convert into numerical
train_df.isna().sum() # check number of NaNs
train_df.describe().T

# %%
# 3) Data Cleaning
# convert into numerical first
train_df['cases_new'] = pd.to_numeric(train_df['cases_new'], errors = 'coerce')
train_df['date'] = train_df['date'].str.replace('/','')
train_df['date'] = train_df['date'].astype(float)
train_df = train_df.drop(['cluster_import'],axis=1)
train_df = train_df.drop(['cluster_religious'],axis=1)
train_df = train_df.drop(['cluster_community'],axis=1)
train_df = train_df.drop(['cluster_highRisk'],axis=1)
train_df = train_df.drop(['cluster_education'],axis=1)
train_df = train_df.drop(['cluster_detentionCentre'],axis=1)
train_df = train_df.drop(['cluster_workplace'],axis=1)

# %%
# Use interpolation method to deal with NaNs/Replace NaNs with value
train_df['cases_new'] = train_df['cases_new'].interpolate(method='polynomial', order=2)
# %% 
# Recheck Data
train_df.info()
# %%
# Visualize Graph
array_df = train_df['cases_new']

plt.figure()
plt.plot(array_df)
plt.show()

# %% 
# 4. Features Selection
open = train_df['cases_new'][::1]

# %%
# 5. Data Preprocessing
mms = MinMaxScaler()
open = mms.fit_transform(open[::,None])
X = []
y = []
win_size = 30
# %%
for i in range (win_size,len(open)):
    X.append(open[i-win_size:i])
    y.append(open[i])

X = np.array(X)
y = np.array(y)

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True, random_state = 123)

# %% 
# 6. Model Development

model = Sequential()
model.add(LSTM(64,return_sequences=True, input_shape=(X_train.shape[1:])))
model.add(Dropout(0.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1,activation='relu'))
model.summary()

plot_model(model, show_shapes=True, show_dtype=True)

model.compile(optimizer='adam', loss =  'mape', metrics = ['mape'])

# %% History
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y&m%d-%H%M%S"))
ts_callback = TensorBoard(log_dir=LOGS_PATH)
es_callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=15,callbacks=[ts_callback,es_callback])

# %%
# 7. Model Analysis
TEST_CSV_PATH = os.path.join(os.getcwd(), 'Test', 'cases_malaysia_test.csv')
test_df= pd.read_csv(TEST_CSV_PATH)
# %%
test_df = test_df.drop(['cluster_import'],axis=1)
test_df = test_df.drop(['cluster_religious'],axis=1)
test_df = test_df.drop(['cluster_community'],axis=1)
test_df = test_df.drop(['cluster_highRisk'],axis=1)
test_df = test_df.drop(['cluster_education'],axis=1)
test_df = test_df.drop(['cluster_detentionCentre'],axis=1)
test_df = test_df.drop(['cluster_workplace'],axis=1)
test_df.info()

# %%
test_df = test_df['cases_new'][::1]
test_df = mms.transform(test_df[:,None])

# %%
# Combine modified and actual
concatenated = np.concatenate((open,test_df))
# %%

plt.figure()
plt.plot(concatenated)
plt.show()


# %%

X_testtest = []
y_testtest = []

for i in range (win_size,len(open)):
    X_testtest.append(concatenated[i-win_size:i])
    y_testtest.append(concatenated[i])

X_testtest = np.array(X_testtest)
y_testtest = np.array(y_testtest)

# %%
predicted = model.predict(X_testtest) # to predict the unseen dataset

# %%
# 8. Model Deployment
plt.figure()
plt.plot(predicted,color='red')
plt.plot(y_testtest,color='blue')
plt.legend(['Predicted','Actual'])
plt.xlabel('Time')
plt.ylabel('Covid Case')
plt.show()

# %%
inversed_actual = mms.inverse_transform(y_testtest)
inversed_prediction = mms.inverse_transform(predicted)

# %%
plt.figure()
plt.plot(inversed_prediction,color='red')
plt.plot(inversed_actual,color='blue')
plt.legend(['Predicted','Actual'])
plt.xlabel('Time')
plt.ylabel('Covid Case')
plt.show()

# %%
# Show mean absolute error
print(mean_squared_error(y_testtest,predicted))
print(mean_absolute_percentage_error(y_testtest,predicted))
print(mean_absolute_error(y_testtest,predicted ))

