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

import params
from merl_brdf_database import merl_brdf


def load_normal_png(n_path):
    """

    :param str n_path:
    :return: N, m, n, mask
    :rtype: (np.ndarray,int, int, np.ndarray)
    """
    n_img = cv2.imread(n_path)[:, :, ::-1]
    m, n, _ = n_img.shape

    N = n_img.reshape(-1, 3).T
    N = N.astype(np.float) / 255. * 2. - 1.
    # N.shape == (3, m * n)

    mask = np.ones(shape=(m * n))
    n_img = n_img.reshape(-1, 3).T
    for i in range(m * n):
        if np.linalg.norm(n_img[:, i]) == 0:
            mask[i] = 0

    return N, m, n, mask


def rendering(N, m, n, mask, L, func_brdf):
    """

    :param np.ndarray N: (3, m * n)
    :param int m:
    :param int n:
    :param np.ndarray mask: (m * n)
    :param np.ndarray L: (light_num, 3)
    :type func_brdf: (light, normal, view) -> (3, )
    :return: Measurement matrix (light_num, 3, m * n)
    :rtype: np.ndarray
    """
    light_num, _ = L.shape
    M = np.zeros(shape=(light_num, 3, m * n))
    view_vec = (0, 0, 1)

    for l in tqdm.tqdm(range(light_num)):
        for i in range(m * n):
            if mask[i] == 0:
                continue

            normal = np.array(N[:, i]).flatten()
            normal /= np.linalg.norm(normal)
            light = np.array(L[l, :]).flatten()
            light /= np.linalg.norm(light)

            rhos = func_brdf(light=light, normal=normal, view=view_vec)
            nl = float(np.dot(light, normal))
            nl = max(0., nl)
            M[l, :, i] = np.array(rhos) * nl

    return M


def output(M, N, m, n, mask, output_path, obj_name, brdf_name):
    """

    :param np.ndarray M:
    :param np.ndarray N:
    :param int m:
    :param int n:
    :param np.ndarray mask:
    :param str output_path: path to dir
    :param str obj_name:
    :param str brdf_name:
    """
    o_path = os.path.join(output_path, obj_name, brdf_name)
    if not os.path.exists(o_path):
        os.makedirs(o_path)

    light_num, _ = L.shape

    assert M.shape == (light_num, 3, m * n)
    assert N.shape == (3, m * n)

    ######
    if N is not None:
        N_img = ((N + 1.) / 2. * 255.).astype(np.uint8)
        if mask is not None:
            N_img[:, mask == 0] = 0
        N_img = N_img.T.reshape(m, n, 3)

        o_path_ = os.path.join(output_path, obj_name, obj_name + ".png")
        cv2.imwrite(o_path_, N_img[:, :, ::-1])
    ######

    def_file_name = "{light_index}.png"
    for l in range(light_num):
        m_img = M[l, :, :]  # [light_num, 3, m*n]
        m_img = m_img.astype(np.float) / np.max(M) * np.iinfo(np.uint16).max
        m_img = m_img.T.reshape(m, n, 3).astype(np.uint16)

        cv2.imwrite(os.path.join(o_path, def_file_name.format(light_index=l)), m_img[:, :, ::-1])


def main(merl_path, n_path, output_dir_path, L):
    """

    :param str merl_path:
    :param str n_path:
    :param str output_dir_path:
    :param np.ndarray L: (light_num, 3)
    :return: None
    """
    print("[*] main: {}, {}".format(merl_path, n_path))

    light_num = len(L)
    assert L.shape == (light_num, 3), L.shape
    assert os.path.exists(merl_path), merl_path
    assert os.path.exists(n_path), n_path

    brdf = merl_brdf.BrdfModel(merl_path)
    N, m, n, mask = load_normal_png(n_path)
    M = rendering(N, m, n, mask, L, brdf)
    output(M, N, m, n, mask, output_path=output_dir_path, obj_name=os.path.basename(n_path), brdf_name=brdf.model_name)


def mix_brdf(brdf1, brdf2):
    def model(light, view, normal):
        # sigma and gamma in Eq. (3)
        sigma = 0.5
        gamma = 0.8
        rvals = sigma * np.array(brdf1(light, view, normal)) + (1 - sigma) * np.array(brdf2(light, view, normal))
        return np.power(rvals, gamma)

    return model


if __name__ == '__main__':
    TRAIN_DATA_PATH = os.path.join(params.DATASET_PATH, "train")
    TEST_DATA_PATH = os.path.join(params.DATASET_PATH, "test")
    L = params.light_source_directions()

    merl_brdf_paths = sorted(glob.glob(params.MERL_DATABASE_PATH))
    n_png_paths = sorted(glob.glob(params.BLOBS_PATH))
    n_png_paths = np.array(n_png_paths, dtype=str)
    #####################
    # test data
    #####################
    for n_path in n_png_paths[[1, 7]]:  # This means blob02 and blob08
        for index1, index2 in [(11, 69), (16, 45), (4, 97), (9, 12)]:  # These pairs are used in Fig. 3.
            brdf1 = merl_brdf.BrdfModel(merl_brdf_paths[index1])
            brdf2 = merl_brdf.BrdfModel(merl_brdf_paths[index2])
            brdf = mix_brdf(brdf1, brdf2)
            N, m, n, mask = load_normal_png(n_path)
            M = rendering(N, m, n, mask, L, brdf)
            output(M, N, m, n, mask, output_path=TEST_DATA_PATH, obj_name=os.path.basename(n_path),
                   brdf_name="{}-{}_{}_{}".format(index1, index2, brdf1.model_name, brdf2.model_name))

    #####################
    # training data
    #####################
    for n_path in n_png_paths[[0, 2, 3, 4, 5, 6, 8, 9]]:
        for merl_path in merl_brdf_paths:
            main(merl_path, n_path, TRAIN_DATA_PATH, L)
