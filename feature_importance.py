import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical

df_list = []

print("Reading csv files")

for gz_file_name in glob.glob(os.path.join("featurized_csvs", "*.csv.gz")):
    df_temp = pd.read_csv(gz_file_name, index_col=None, header=0)
    df_list.append(df_temp)

print("Reading done")

dataset = pd.concat(df_list, axis=0, ignore_index=True)

#dataset = pd.read_csv("data.csv")

boolean_columns = ["discrete:app_state:is_active", "discrete:app_state:is_inactive", "discrete:app_state:is_background",
                   "discrete:app_state:missing", "discrete:battery_plugged:is_ac", "discrete:battery_plugged:is_usb",
                   "discrete:battery_plugged:is_wireless", "discrete:battery_plugged:missing",
                   "discrete:battery_state:is_unknown", "discrete:battery_state:is_unplugged",
                   "discrete:battery_state:is_not_charging", "discrete:battery_state:is_discharging",
                   "discrete:battery_state:is_charging", "discrete:battery_state:is_full",
                   "discrete:battery_state:missing", "discrete:on_the_phone:is_False", "discrete:on_the_phone:is_True",
                   "discrete:on_the_phone:missing", "discrete:ringer_mode:is_normal",
                   "discrete:ringer_mode:is_silent_no_vibrate", "discrete:ringer_mode:is_silent_with_vibrate",
                   "discrete:ringer_mode:missing", "discrete:wifi_status:is_not_reachable",
                   "discrete:wifi_status:is_reachable_via_wifi", "discrete:wifi_status:is_reachable_via_wwan",
                   "discrete:wifi_status:missing"]
                   # "watch_acceleration:magnitude_stats:mean",	"watch_acceleration:magnitude_stats:std",
                   # "watch_acceleration:magnitude_stats:moment3",	"watch_acceleration:magnitude_stats:moment4",
                   # "watch_acceleration:magnitude_stats:percentile25",	"watch_acceleration:magnitude_stats:percentile50",
                   # "watch_acceleration:magnitude_stats:percentile75",	"watch_acceleration:magnitude_stats:value_entropy",
                   # "watch_acceleration:magnitude_stats:time_entropy",	"watch_acceleration:magnitude_spectrum:log_energy_band0",
                   # "watch_acceleration:magnitude_spectrum:log_energy_band1",	"watch_acceleration:magnitude_spectrum:log_energy_band2",
                   # "watch_acceleration:magnitude_spectrum:log_energy_band3",	"watch_acceleration:magnitude_spectrum:log_energy_band4",
                   # "watch_acceleration:magnitude_spectrum:spectral_entropy",	"watch_acceleration:magnitude_autocorrelation:period",
                   # "watch_acceleration:magnitude_autocorrelation:normalized_ac",	"watch_acceleration:3d:mean_x",
                   # "watch_acceleration:3d:mean_y",	"watch_acceleration:3d:mean_z",	"watch_acceleration:3d:std_x",
                   # "watch_acceleration:3d:std_y",	"watch_acceleration:3d:std_z",	"watch_acceleration:3d:ro_xy",	"watch_acceleration:3d:ro_xz",
                   # "watch_acceleration:3d:ro_yz",	"watch_acceleration:spectrum:x_log_energy_band0",
                   # "watch_acceleration:spectrum:x_log_energy_band1",	"watch_acceleration:spectrum:x_log_energy_band2",
                   # "watch_acceleration:spectrum:x_log_energy_band3",	"watch_acceleration:spectrum:x_log_energy_band4",
                   # "watch_acceleration:spectrum:y_log_energy_band0",	"watch_acceleration:spectrum:y_log_energy_band1",
                   # "watch_acceleration:spectrum:y_log_energy_band2",	"watch_acceleration:spectrum:y_log_energy_band3",
                   # "watch_acceleration:spectrum:y_log_energy_band4",	"watch_acceleration:spectrum:z_log_energy_band0",
                   # "watch_acceleration:spectrum:z_log_energy_band1",	"watch_acceleration:spectrum:z_log_energy_band2",
                   # "watch_acceleration:spectrum:z_log_energy_band3",	"watch_acceleration:spectrum:z_log_energy_band4",
                   # "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0",
                   # "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1",
                   # "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2",
                   # "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3",
                   # "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4"]

print("Cleaning and preparing dataframe")

dataset = dataset.fillna(0)
dataset[boolean_columns] = (dataset[boolean_columns] == 'TRUE').astype(int)
dataset = dataset.select_dtypes(np.number)

X = dataset.drop(['label:SLEEPING'], axis=1)
y = dataset['label:SLEEPING']

# normalizing
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)

print("Dataframe ready")

model = XGBClassifier()
model.fit(X, y)

print("Top 10 most important features:")
df_feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['feature importance']).sort_values(
    'feature importance', ascending=False)
print(df_feature_imp.iloc[:10, ])

print("Dimensions before numpy and reshape")
print(dataset.shape)
print(X.shape)
print(y.shape)

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=0)

# trainX = np.expand_dims(trainX.values, axis=0)
# trainy = np.expand_dims(trainy.values, axis=0)
# testX = np.expand_dims(testX.values, axis=0)
# testy = np.expand_dims(testy.values, axis=0)

xgb_predict = model.predict(testX)
xgb_cv_score = cross_val_score(model, X, y, cv=10, scoring='roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(testy, xgb_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(testy, xgb_predict))
print('\n')
print("=== All AUC Scores ===")
print(xgb_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - XGB: ", xgb_cv_score.mean())

#
# # convert series to supervised learning
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
#
#
# values = dataset.values
# # ensure all data is float
# values = values.astype('float32')
#
# # moving sleep column
# sleep_column = values[:, 210]
# values = np.delete(values, 210, axis=1)
# values = np.column_stack((values, sleep_column))
#
# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
#
# # frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
# # drop columns we don't want to predict
# reframed.drop(reframed.columns[[]],
#               axis=1, inplace=True)
# print(reframed.head())
#
# # split into train and test sets
# values = reframed.values
#
# y = values[:, -1]
# X = np.delete(values, -1, axis=1)
#
# print(X.shape)
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
#
# # # split into input and outputs
# # train_X, train_y = train[:, :-1], train[:, -1]
# # test_X, test_y = test[:, :-1], test[:, -1]
#
# # reshape input to be 3D [samples, timesteps, features]
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#
# # design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
#                     shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
#
# # make a prediction
# yhat = model.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
#
# #
# # # fit and evaluate a model
# # def evaluate_model(trainX, trainy, testX, testy):
# #     verbose, epochs, batch_size = 0, 15, 64
# #     n_timesteps, n_features= trainX.shape[1], trainX.shape[2]
# #     n_outputs = 1
# #     print("timesteps: ")
# #     print(n_timesteps)
# #     print(n_features)
# #     print(n_outputs)
# #     model = Sequential()
# #     model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
# #     model.add(Dropout(0.5))
# #     model.add(Dense(100, activation='relu'))
# #     model.add(Dense(n_outputs, activation='sigmoid'))
# #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #     # fit network
# #     model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# #     # evaluate model
# #     _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
# #     return accuracy
# #
# #
# # # summarize scores
# # def summarize_results(scores):
# #     print(scores)
# #     m, s = mean(scores), std(scores)
# #     print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# #
# #
# # # run an experiment
# # def run_experiment(repeats=10):
# #     # load data
# #     trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=0)
# #
# #     trainX = np.expand_dims(trainX.values, axis=0)
# #     trainy = np.expand_dims(trainy.values, axis=0)
# #     testX = np.expand_dims(testX.values, axis=0)
# #     testy = np.expand_dims(testy.values, axis=0)
# #
# #     # repeat experiment
# #     scores = list()
# #     for r in range(repeats):
# #         score = evaluate_model(trainX, trainy, testX, testy)
# #         score = score * 100.0
# #         print('>#%d: %.3f' % (r + 1, score))
# #         scores.append(score)
# #     # summarize results
# #     summarize_results(scores)
# #
# #
# # run_experiment(1)
