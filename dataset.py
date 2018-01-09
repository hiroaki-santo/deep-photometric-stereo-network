#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import cv2
import numpy as np
import tqdm


class DpsnDataset(object):
    def __init__(self, dataset_path="./dataset/", name="blobs_merl"):
        self.dataset_path = dataset_path
        self.data_list = glob.glob(os.path.join(self.dataset_path, "*/", "*/"))
        self.name = name

        tmp_list = glob.glob(os.path.join(self.data_list[0], "[0-9]*.png"))
        self.light_num = len(tmp_list)
        self.img_size = cv2.imread(tmp_list[0])[:, :, 0].shape

        print("light_num: {}".format(self.light_num))
        print("image_size: {}".format(self.img_size))
        print("data_num: {}".format(len(self)))

        np.random.seed(1)
        self.random_indices = np.random.permutation(len(self))

    def data_path2name(self, path):
        dir_path, _ = os.path.split(path)
        dir_path, brdf_name = os.path.split(dir_path)
        _, obj_name = os.path.split(dir_path)

        return obj_name, brdf_name

    def __load_normal_png(self, n_path):
        n_img = cv2.imread(n_path)[:, :, ::-1]
        m, n, _ = n_img.shape

        N = n_img.reshape(-1, 3).T
        N = N.astype(np.float32) / 255. * 2. - 1.
        for i in range(m * n):
            norm = np.linalg.norm(N[:, i])
            if norm != 0:
                N[:, i] /= norm

        mask = np.ones(shape=(m * n))
        n_img = n_img.reshape(-1, 3).T
        for i in range(m * n):
            if np.linalg.norm(n_img[:, i]) == 0:
                mask[i] = 0

        N[:, mask == 0] = 0
        return N, m, n, mask

    def __len__(self):
        return len(self.data_list)

    def load_data(self, root_path):
        def_png_path = os.path.join(root_path, "{light_index}.png")

        m, n = self.img_size
        M = np.zeros(shape=(m * n, self.light_num, 3), dtype=np.float32)
        for l in range(self.light_num):
            m_img = cv2.imread(def_png_path.format(light_index=l), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
            # m_img = cv2.imread(def_png_path.format(light_index=l))[:, :, ::-1]
            M[:, l, :] = m_img.reshape(-1, 3)

        obj_name, brdf_name = self.data_path2name(root_path + "/")
        N, m, n, mask = self.__load_normal_png(os.path.join(self.dataset_path, obj_name, "{}.png".format(obj_name)))

        return M, N, mask

    def get_batch(self, index, image_num):
        """

        :param index:
        :param image_num: This does not mean the number of batch data. Number of pixels in image_num images becomes number of data.
        :return:
        """

        indices = np.arange(index, index + image_num)
        indices %= len(self)
        indices = self.random_indices[indices]

        batch_normal = []
        batch_mess = []
        for i in indices:
            M, N, mask = self.load_data(self.data_list[i])
            M = M.astype(float) / np.max(M)
            for p in range(N.shape[1]):
                if mask[p] == 0:
                    continue
                if np.min(np.linalg.norm(M[p, :, :], axis=0)) == 0:
                    continue
                for color in range(3):
                    batch_normal.append(N[:, p])
                    batch_mess.append(M[p, :, color])

        return np.array(batch_normal, dtype=np.float32), np.array(batch_mess, dtype=np.float32)

    def save_as_tfrecord(self):
        print("[*] save_as_tfrecord()")
        import tensorflow as tf

        tfwriter = tf.python_io.TFRecordWriter(
            os.path.join(self.dataset_path, "{}_{}.tfrecord".format(type(self).__name__, self.name)))

        try:
            for i in tqdm.tqdm(range(0, len(self), 30)):
                normal, mess = self.get_batch(index=i, image_num=30)
                print("serialize data: {}, {}".format(normal.shape, mess.shape))
                for j in np.random.permutation(len(normal)):
                    n_ = normal[j, :].astype(np.float32)
                    m_ = mess[j, :].astype(np.float32)
                    record = tf.train.Example(features=tf.train.Features(feature={
                        'normal': tf.train.Feature(float_list=tf.train.FloatList(value=n_.reshape(-1).tolist())),
                        'mess': tf.train.Feature(
                            float_list=tf.train.FloatList(value=m_.reshape(-1).tolist())),
                    }))
                    tfwriter.write(record.SerializeToString())
        finally:
            tfwriter.close()

    def load_from_tfrecord(self):
        import tensorflow as tf

        data_path = os.path.join(self.dataset_path, "{}_{}.tfrecord".format(type(self).__name__, self.name))
        assert os.path.exists(data_path), data_path

        filename_queue = tf.train.string_input_producer([data_path], num_epochs=None)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'normal': tf.FixedLenFeature([3], tf.float32),
                'mess': tf.FixedLenFeature([self.light_num], tf.float32),
            })
        return features["normal"], features["mess"]


if __name__ == '__main__':
    import params

    dataset = DpsnDataset(dataset_path=os.path.join(params.DATASET_PATH, "train"))
    dataset.save_as_tfrecord()
