# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 18_July_2020
# Toy Model.
#
# For GOD I Trust.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

class toyModel(object):
  """A Toy Model."""
  def __init__(self, input):
    self.output = self.forward(input)
  
  def forward(self, input):
    with tf.variable_scope('linear'):
      output = tf.layers.dense(input,
                               2,
                               activation=None,
                               name='final_linear',
                               kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2))
    return output
  
  def getOutput(self):
    return self.output