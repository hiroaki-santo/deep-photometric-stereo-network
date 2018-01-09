#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import matplotlib
import numpy as np
import tensorflow as tf
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_float("shadow_cast_prob", 0.05, "")

tf.app.flags.DEFINE_integer("saver_keep_num", 10, "")
tf.app.flags.DEFINE_boolean("load_ckpt", True, "")
tf.app.flags.DEFINE_integer("ckpt_interval_epoch", 1000, "")
tf.app.flags.DEFINE_integer("eval_output_image_num", 2, "")
FLAGS = tf.app.flags.FLAGS


class DpsnModel(object):
    def __init__(self, sess, output_path, light_num):
        self.sess = sess
        self.output_path = output_path
        self.light_num = light_num

        self.checkpoint_dir = os.path.join(output_path, "checkpoint")
        self.best_checkpoint_dir = os.path.join(output_path, "best_checkpoint")
        self.model_name = self.__class__.__name__ + ".ckpt"

        self.summary_train = []
        self.summary_test = []

        self.build_model()

        # this must be called after all variables defined.
        self.best_saver = tf.train.Saver(max_to_keep=1)
        self.saver = tf.train.Saver(max_to_keep=FLAGS.saver_keep_num)

        summary_path = os.path.join(self.output_path, "summary")
        self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)

    def build_model(self):
        self.mess = tf.placeholder(tf.float32, [None, self.light_num], name="mess")
        self.normal_ = tf.placeholder(tf.float32, [None, 3], name="normal_")
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.shadow_keep_prob = tf.placeholder_with_default(input=self.keep_prob, shape=[])

        drop_rate = 1 - self.keep_prob
        shadow_drop_rate = 1 - self.shadow_keep_prob

        ##############
        normal_ = self.normal_
        mess = self.mess
        norm = tf.norm(mess, axis=1, keep_dims=True)
        norm = tf.tile(norm, [1, self.light_num])
        norm = tf.where(tf.is_nan(norm), tf.ones_like(norm), norm)
        mess = tf.divide(mess, norm)

        with tf.name_scope("mlp") as scope:
            net = tf.layers.dropout(mess, rate=shadow_drop_rate)
            net = tf.scalar_mul(1. - shadow_drop_rate, net)

            net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name='dense1')
            net = tf.layers.dropout(net, rate=drop_rate)
            net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name='dense2')
            net = tf.layers.dropout(net, rate=drop_rate)
            net = tf.layers.dense(net, 2048, activation=tf.nn.relu, name='dense3')
            net = tf.layers.dropout(net, rate=drop_rate)
            net = tf.layers.dense(net, 2048, activation=tf.nn.relu, name='dense4')
            net = tf.layers.dropout(net, rate=drop_rate)
            net = tf.layers.dense(net, 2048, activation=tf.nn.relu, name='dense5')
            net = tf.layers.dropout(net, rate=drop_rate)
            self.normal = tf.layers.dense(net, 3, activation=None, name='danse6')

        with tf.name_scope("train") as scope:
            self.cost = tf.reduce_mean(tf.square(normal_ - self.normal))
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)
            self.summary_train.append(tf.summary.scalar("cost", self.cost))

            self.eval_RMSE = tf.reduce_mean(tf.norm(normal_ - self.normal, axis=1))
            self.summary_train.append(tf.summary.scalar("RMSE", self.eval_RMSE))

        with tf.name_scope("test") as scope:
            self.summary_test.append(tf.summary.scalar("cost", self.cost))
            self.summary_test.append(tf.summary.scalar("RMSE", self.eval_RMSE))

    def _shadow_drop_keep_rate(self):
        x = np.random.binomial(self.light_num, FLAGS.shadow_cast_prob)
        x = float(x) / self.light_num
        x = np.clip(x, 0., 1.)

        return 1. - x

    def def_feed(self, train=True):
        if train:
            feed = {self.keep_prob: 0.5, self.shadow_keep_prob: self._shadow_drop_keep_rate()}
        else:
            feed = {self.keep_prob: 1.0, self.shadow_keep_prob: 1.0}

        return feed

    def train(self, train_dataset, test_dataset, batch_size, epoch_num):
        if FLAGS.load_ckpt:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)

            if could_load:
                print(" [*] Load SUCCESS")
                epoch_start = checkpoint_counter
            else:
                print(" [!] Load failed...")
                print(" [*] Init variables")
                epoch_start = 0
                self.sess.run(tf.global_variables_initializer())
        else:
            print(" [*] Init variables")
            epoch_start = 0
            self.sess.run(tf.global_variables_initializer())

        train_normal, train_mess = train_dataset.load_from_tfrecord()
        test_normal, test_mess = test_dataset.get_batch(0, len(test_dataset))

        train_normal, train_mess = tf.train.shuffle_batch([train_normal, train_mess],
                                                          batch_size=batch_size,
                                                          capacity=200 * 200 * 8 * 100 * 3,
                                                          min_after_dequeue=200 * 200 * 16,
                                                          num_threads=7)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            best_cost = np.finfo(np.float32).max

            print("[*] start training...")
            for epoch in tqdm.tqdm(range(epoch_start + 1, epoch_num)):
                batch_normal, batch_mess = self.sess.run([train_normal, train_mess])
                feed = {self.mess: batch_mess, self.normal_: batch_normal}
                feed.update(self.def_feed(train=True))
                self.train_step.run(feed_dict=feed)
                #####
                if epoch % 100 == 0:
                    feed = {self.mess: batch_mess, self.normal_: batch_normal}
                    feed.update(self.def_feed(train=False))
                    summary_str = self.sess.run(tf.summary.merge(self.summary_train), feed_dict=feed)
                    self.summary_writer.add_summary(summary_str, epoch)
                    #
                    indices = np.random.permutation(len(test_mess))
                    feed = {self.mess: test_mess[indices[0: batch_size]],
                            self.normal_: test_normal[indices[0: batch_size]]}
                    feed.update(self.def_feed(train=False))

                    summary_str = self.sess.run(tf.summary.merge(self.summary_test), feed_dict=feed)
                    self.summary_writer.add_summary(summary_str, epoch)
                    self.summary_writer.flush()

                    cost = self.cost.eval(feed_dict=feed)
                    if cost < best_cost:
                        print("{}, best cost: {} => {}".format(epoch, best_cost, cost))
                        self.best_save(step=epoch)
                        best_cost = cost
                    else:
                        print("{}, best cost: {} (now: {})".format(epoch, best_cost, cost))

                ##############
                # save
                ##############
                if epoch % FLAGS.ckpt_interval_epoch == 0:
                    self.save(step=epoch)
                    self.eval(test_dataset, num=FLAGS.eval_output_image_num, step=epoch)

        finally:
            coord.request_stop()
            coord.join(threads)

    def test(self, M, m, n, batch_size):
        assert M.shape == (m * n, self.light_num, 3)

        indices = np.arange(m * n)
        N = np.zeros(shape=(m * n, 3, 3))
        for color in range(3):
            for index in np.array_split(indices, max(1, int(m * n / batch_size))):
                mess = M[index, :, color]
                feed = {self.mess: mess}
                feed.update(self.def_feed(train=False))
                normal = self.sess.run(self.normal, feed_dict=feed)
                N[index, color, :] = normal

        for i in range(m * n):
            for c in range(3):
                N[i, c, :] /= np.linalg.norm(N[i, c, :])
        N = np.average(N, axis=1).T

        for i in range(m * n):
            N[:, i] /= np.linalg.norm(N[:, i])
        return N

    def eval(self, dataset, num, step):
        o_path = os.path.join(self.output_path, "eval")
        if not os.path.exists(o_path):
            os.makedirs(o_path)

        m, n = dataset.img_size
        for i in range(num):
            M, N_, mask = dataset.load_data(dataset.data_list[i])
            M /= np.max(M)
            N = self.test(M, m, n, 10000)

            obj_name, brdf_name = dataset.data_path2name(dataset.data_list[i])

            MAE = np.zeros(shape=(m * n))
            for j in range(m * n):
                if mask[j] == 0:
                    continue
                error = np.dot(N_[:, j], N[:, j]) / (np.linalg.norm(N_[:, j]) * np.linalg.norm(N[:, j]))
                error = np.arccos(np.clip(error, -1., 1.))
                error = np.rad2deg(error)
                MAE[j] = error

            N[:, mask == 0] = 0
            n_img = (N.T.reshape(m, n, 3) + 1.) / 2. * 255.
            plt.figure()
            plt.imshow(n_img.astype(np.uint8))
            plt.title("MAE: {}".format(np.average(MAE[mask != 0])))
            plt.savefig(os.path.join(o_path, "{}_{}-{}.png").format(step, obj_name, brdf_name))
            plt.close()

    def save(self, step):
        try:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            save_path = os.path.join(self.checkpoint_dir, self.model_name)
            self.saver.save(self.sess, save_path, write_meta_graph=False, global_step=step)
        except Exception as e:
            print("[!] error save...")
            print(e.message)

    def best_save(self, step):
        try:
            if not os.path.exists(self.best_checkpoint_dir):
                os.makedirs(self.best_checkpoint_dir)

            save_path = os.path.join(self.best_checkpoint_dir, self.model_name)
            self.best_saver.save(self.sess, save_path, write_meta_graph=False, global_step=step)
        except Exception as e:
            print("[!] error save...")
            print(e.message)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Loading checkpoints... {}".format(checkpoint_dir))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to restore {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
