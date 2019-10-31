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
early_stopping_delta = 1e-4   # How much the CRPS must improve before stopping
steps_max = 1000   # How many steps to fit at max
lr = np.asarray(0.1, dtype='float32')   # The learning rate


# for single day
date_str = '2019-05-04-12'
train_set, test_set = get_train_test_sets(DATA_DIR, predict_date=date_str,
                                          fclt=fclt, window_size=window_size,
                                          full_ensemble_t=False)
batch_size = train_set.features.shape[0]

# for day loops
date_str_start = '2019-01-03-00'
date_str_stop = '2019-07-18-12'

# FC Network
# fc_model = build_fc_model(2, 2, compile=True, optimizer='sgd')
# fc_model.summary()
#
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
# result = [1.98311387366, 1.98747289806]

# CNN Network

cnn_model = build_cnn_model(train_set.features.shape[1:], compile=True, optimizer='sgd')
cnn_model.summary()

cnn_model.fit(train_set.features, train_set.targets, epochs=steps_max,
             batch_size=batch_size,
             validation_data=[test_set.features, test_set.targets],
             verbose=0,
             callbacks=[EarlyStopping(monitor='loss',
                                      min_delta=early_stopping_delta,
                                      patience=2)])

print(cnn_model.evaluate(train_set.features, train_set.targets, batch_size, verbose=0),
 cnn_model.evaluate(test_set.features, test_set.targets, batch_size, verbose=0))
# result = [1.98311387366, 1.98747289806]

# EMOS Network
# model_keras = build_EMOS_network_keras(compile=True, optimizer='sgd', lr=0.1)
# model_keras.summary()
#
# train_mean = train_set.features[:, 0]
# train_std = train_set.features[:, 1]
# test_mean = test_set.features[:, 0]
# test_std = test_set.features[:, 1]
#
# model_keras.fit([train_mean, train_std], train_set.targets, epochs=steps_max,
#                 batch_size=batch_size,
#                 validation_data=[[test_mean, test_std], test_set.targets],
#                 verbose=0,
#                 callbacks=[EarlyStopping(monitor='loss',
#                                          min_delta=early_stopping_delta,
#                                          patience=2)]);
#
# print(model_keras.evaluate([train_mean, train_std], train_set.targets, batch_size, verbose=0),
#  model_keras.evaluate([test_mean, test_std], test_set.targets, batch_size, verbose=0))

# Embedding with hidden layer
# max_id = int(np.max([train_set.cont_ids.max(), test_set.cont_ids.max()]))
#
# def build_and_run_emb_model(emb_size):
#     emb_model = build_emb_model(2, 2, [], emb_size, max_id, compile=True)
#     emb_model.fit([train_set.features, train_set.cont_ids], train_set.targets,
#                   epochs=40,batch_size=1024, verbose=0,
#                   validation_data=[[test_set.features, test_set.cont_ids], test_set.targets])
#     print(emb_model.evaluate([train_set.features, train_set.cont_ids], train_set.targets, verbose=0),
#           emb_model.evaluate([test_set.features, test_set.cont_ids], test_set.targets, verbose=0))
#
# build_and_run_emb_model(5)

# Run for multidays
# max_id = int(np.max([train_set.cont_ids.max(), test_set.cont_ids.max()]))
# emb_model = build_emb_model(2, 2, [], 5, max_id, compile=True)
#
# train_crps_list, valid_crps_list, results_df = loop_over_days(
#     DATA_DIR,
#     emb_model,
#     date_str_start, date_str_stop,
#     window_size=window_size,
#     fclt=fclt,
#     epochs_max=steps_max,
#     early_stopping_delta=early_stopping_delta,
#     lr=0.1,
#     verbose=0,
#     model_type='embedding')
# #%%
#
# print(np.mean(train_crps_list), np.mean(valid_crps_list))
# [1.31448369234, 1.43893041276]
# preds = emb_model.predict([test_set.features, test_set.cont_ids])
# results_df = create_results_df(test_set.date_strs, test_set.station_ids,
#                                preds[:, 0], preds[:, 1])
# results_df.to_csv(results_dir + 'embedding_fc_train_2015_pred_2016.csv')
