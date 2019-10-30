import sys
sys.path.append('../')   # This is where all the python files are!
from importlib import reload
from nn_src.emos_network_theano import EMOS_Network
from nn_src.losses import crps_cost_function
from nn_src.utils import *
from nn_src.keras_models import *
from collections import OrderedDict

# Basic setup
DATA_DIR = '/data/gaojinghan/EC_ensemble/manipulated/'  # Mac
# DATA_DIR = '/project/meteo/w2w/C7/ppnn_data/'   # LMU
results_dir = './'
window_size = 25   # Days in rolling window
fclt = 0   # Forecast lead time in hours


# date_str = '2019-05-04-12'
# train_set, test_set = get_train_test_sets(DATA_DIR, predict_date=date_str,
#                                           fclt=fclt, window_size=window_size,
#                                           full_ensemble_t=False)
#
# fc_model = build_fc_model(2, 2, compile=True, optimizer='sgd')
# fc_model.summary()

early_stopping_delta = 1e-4   # How much the CRPS must improve before stopping
steps_max = 1000   # How many steps to fit at max
lr = np.asarray(0.1, dtype='float32')   # The learning rate
# batch_size = train_set.features.shape[0]

# fc_model.fit(train_set.features, train_set.targets, epochs=steps_max,
#              batch_size=batch_size,
#              validation_data=[test_set.features, test_set.targets],
#              verbose=0,
#              callbacks=[EarlyStopping(monitor='loss',
#                                       min_delta=early_stopping_delta,
#                                       patience=2)]);
#
# print(fc_model.evaluate(train_set.features, train_set.targets, batch_size, verbose=0),
#  fc_model.evaluate(test_set.features, test_set.targets, batch_size, verbose=0))

date_str_start = '2019-01-03-00'
date_str_stop = '2019-07-18-12'

# fc_model = build_fc_model(2, 2, compile=True)
#
# train_crps_list, valid_crps_list, results_df = loop_over_days(
#     DATA_DIR,
#     fc_model,
#     date_str_start, date_str_stop,
#     window_size=window_size,
#     fclt=fclt,
#     epochs_max=steps_max,
#     early_stopping_delta=early_stopping_delta,
#     lr=0.1,
#     verbose=0)
#
# print(np.mean(train_crps_list), np.mean(valid_crps_list))
# result = [1.95584055702, 1.96096154518]
#
# results_df.to_csv(results_dir + 'fc_network_rolling_window.csv')

# model_keras = build_EMOS_network_keras(compile=True, optimizer='adam', lr=0.1)
#
# train_crps_list, valid_crps_list, results_df = loop_over_days(
#     DATA_DIR,
#     model_keras,
#     date_str_start, date_str_stop,
#     window_size=window_size,
#     fclt=fclt,
#     epochs_max=steps_max,
#     early_stopping_delta=early_stopping_delta,
#     lr=lr,
#     verbose=0,
#     model_type='EMOS_Network_keras')
#
# print(np.mean(train_crps_list), np.mean(valid_crps_list))
#
# result = [1.98311387366, 1.98747289806]
#
# results_df.to_csv(results_dir + 'emos_network_rolling_window_keras.csv')

emb_size = 3
max_id = int(np.max([train_set.cont_ids.max(), test_set.cont_ids.max()]))

#%%

emb_model = build_emb_model(2, 2, [], emb_size, max_id, compile=True,
                            lr=0.01)

#%%

emb_model.summary()

#%% md

### Train 2015, predict 2016

#%%

# Ran this for 40 epochs

train_crps_list, valid_crps_list, results_df = loop_over_days(
    DATA_DIR,
    emb_model,
    date_str_start, date_str_stop,
    window_size=window_size,
    fclt=fclt,
    epochs_max=steps_max,
    early_stopping_delta=early_stopping_delta,
    lr=0.1,
    verbose=0)
#%%

preds = emb_model.predict([test_set.features, test_set.cont_ids])
results_df = create_results_df(test_set.date_strs, test_set.station_ids,
                               preds[:, 0], preds[:, 1])
results_df.to_csv(results_dir + 'embedding_fc_train_2015_pred_2016.csv')
