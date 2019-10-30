from .keras_models import *
from keras.utils.generic_utils import get_custom_objects
metrics_dict = dict([(f.__name__, f) for f in [crps_cost_function]])
get_custom_objects().update(metrics_dict)
import tensorflow as tf

def limit_mem():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
limit_mem()