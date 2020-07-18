# coding:utf-8

import sys
import copy
import random
import codecs
import pickle
import logging
import functools
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(MAIN_PATH))
from tensorflow.contrib import predictor
from model import toyModel

tf.compat.v1.logging.set_verbosity(logging.INFO)
logging.StreamHandler(sys.stdout)

def getBatchIndex(data_length, batch_size):
  batch_num = data_length // batch_size
  batch_num = batch_num if data_length % batch_size == 0 \
    else batch_num + 1
  
  for i in range(batch_num):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    yield (start_idx, end_idx)

def dataGenerator(path, batch_size):
  with codecs.open(path, 'rb') as file:
    data = pickle.load(file)
  data_length = len(data)
  data = copy.deepcopy(data)
  random.shuffle(data)

  for (start, end) in getBatchIndex(data_length, batch_size):
    data_batch = data[start:end]
    input_data = [pair[0] for pair in data_batch]
    golden_y = [pair[1] for pair in data_batch]
    
    features = {'input': input_data}

    yield (features, golden_y)
    
def trainInputFn(path, batch_size, train_steps):
  output_types = {'input': tf.float32}
  output_shapes = {'input': [None, 10]}

  dataGenerator_args = functools.partial(dataGenerator, path=path, batch_size=batch_size)
  dataset = tf.data.Dataset.from_generator(
    dataGenerator_args,
    output_types=(output_types, tf.int32),
    output_shapes=(output_shapes, [None]))
  
  dataset = dataset.repeat(train_steps)

  return dataset

def servingInputFn():
  input_data = tf.placeholder(tf.float32, shape=[None, 10], name='input')
  receive_tensor = {'input': input_data}
  features = {'input': input_data}

  return tf.estimator.export.ServingInputReceiver(features, receive_tensor)

def createData(is_save=True):
  data = []
  for _ in range(1000):
    seed = random.randint(0, 1)
    if seed == 0:
      data_s = np.random.uniform(low=10.0, high=20.0, size=(10, ))
    else:
      data_s = np.random.uniform(low=1.0, high=10.0, size=(10, ))
    data.append((data_s, seed))
  
  if is_save:
    with codecs.open('sample_data.bin', 'wb') as file:
      pickle.dump(data, file)
  else:
    return data

def modelFnBuilder():
  def modelFn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    input_data = features['input']
    model = toyModel(input_data)
    output = model.getOutput()
    batch_size = tf.cast(tf.shape(output)[0], dtype=tf.float32)

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {'output': output}
      output_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
      loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=output)) / batch_size

      learning_rate = tf.constant(1e-2, dtype=tf.float32)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      tvars = tf.trainable_variables()
      gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=True)
      clipped_gradients, _  = tf.clip_by_global_norm(gradients, 5.0)
      train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=tf.train.get_global_step())

      logging_hook = tf.train.LoggingTensorHook({'step': tf.train.get_global_step(),
                                                 'loss': loss}, every_n_iter=1)
      output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

    return output_spec
  
  return modelFn
      
def main():
  Path('../../models/toy_model').mkdir(exist_ok=True)

  model_fn = modelFnBuilder()
  run_config = tf.estimator.RunConfig(
    keep_checkpoint_max=1,
    save_checkpoints_steps=10,
    model_dir='../../models/toy_model')
  
  estimator = tf.estimator.Estimator(model_fn, config=run_config)
  train_input_fn_args = functools.partial(trainInputFn, path='sample_data.bin', batch_size=10, train_steps=10)
  estimator.train(train_input_fn_args)

def packageModel():
  model_fn = modelFnBuilder()
  estimator = tf.estimator.Estimator(model_fn, '../../models/toy_model')
  estimator.export_saved_model('./app_models/', servingInputFn)

def restoreModel(pb_path):
  subdirs = [x for x in Path(pb_path).iterdir()
    if x.is_dir() and 'temp' not in str(x)]
  latest_model = str(sorted(subdirs)[-1])
  model = predictor.from_saved_model(latest_model)
  
  return model

if __name__ == '__main__':
  # main()
  model = restoreModel('./app_models/')
  test_data = np.random.uniform(low=1.0, high=10.0, size=(1, 10))
  features = {'input': test_data}
  output = model(features)['output'][0]
  output_prob = tf.nn.softmax(output)
  predict_id = 1 if output_prob[1] > output_prob[0] else 0
  print(predict_id)