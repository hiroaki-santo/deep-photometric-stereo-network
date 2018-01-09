#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import params
from dataset import DpsnDataset
from model import DpsnModel

tf.app.flags.DEFINE_string("dataset_path", params.DATASET_PATH, "")
tf.app.flags.DEFINE_string("output_path", "./model", "")
tf.app.flags.DEFINE_integer("batch_size", 1000, "")
tf.app.flags.DEFINE_integer("steps", 40000, "")
tf.app.flags.DEFINE_integer("gpu", 0, "gpu id")

FLAGS = tf.app.flags.FLAGS

tf_config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="{}".format(FLAGS.gpu),
        allow_growth=True,
    )
)


def main(_):
    train_dataset = DpsnDataset(dataset_path=os.path.join(FLAGS.dataset_path, "train"))
    test_dataset = DpsnDataset(dataset_path=os.path.join(FLAGS.dataset_path, "test"))

    with tf.Session(config=tf_config) as sess:
        model = DpsnModel(sess=sess, output_path=FLAGS.output_path, light_num=train_dataset.light_num)
        model.train(train_dataset, test_dataset, FLAGS.batch_size, FLAGS.steps)


if __name__ == '__main__':
    tf.app.run(main=main)
