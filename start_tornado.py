# coding:utf-8

import numpy as np
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import json
import tornado.ioloop
import tornado.web
import tornado.escape

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

class ToyModelServer(tornado.web.RequestHandler):
  def initialize(self):
    server_url = 'localhost:8500'
    channel = grpc.insecure_channel(server_url)
    self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    self.tf_request = predict_pb2.PredictRequest()
    self.tf_request.model_spec.name = 'toyModel'  # 模型名称
    # self.tf_request.model_spec.signature_name = 'toyModel_tf_serving'  # 签名名称

  def prepare(self):
    print('prepare...')
    data = tornado.escape.json_decode(self.request.body)
    text = data.get('text', '')
    self.input_x = self.preprocess_input(text).astype(np.float32)
  
  def preprocess_input(self, input_data):
    print('input_data: ', type(input_data))
    return np.random.uniform(low=1.0, high=10.0, size=(1, 10)) if input_data is '1' \
      else np.random.uniform(low=10.0, high=20.0, size=(1, 10))
  
  def process_output(self, output_data):
    return np.argmax(softmax(output_data))

  def get(self):
    print('GET request.')
    self.write('GET request!')

  def post(self):
    print('POST request.')

    self.tf_request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(self.input_x, shape=[1, 10]))
    tf_response = self.stub.Predict(self.tf_request, 5.0)  # 5 secs timeout
    # print(tf_response.outputs['prediction'])
    y_pred = list(tf_response.outputs['output'].float_val)
    y_pred = int(self.process_output(y_pred))
    print('### Y_PRED ###: ', y_pred)

    code = 0
    result = {
        "code": code,
        # "input_text": text,
        "prediction": y_pred
    }
    self.write(json.dumps(result, ensure_ascii=False))

def make_app():
  return tornado.web.Application([
      (r"/toy_model", ToyModelServer)
  ])

if __name__ == '__main__':
  app = make_app()
  app.listen(9898)
  tornado.ioloop.IOLoop.current().start()