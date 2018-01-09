#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np

NORMAL_MAP_PNG_FILE_NAME = 'Normal_gt.png'
NORMAL_MAP_TEXT_FILE_NAME = 'normal.txt'
LIGHT_DIRECTIONS_FILE_NAME = 'light_directions.txt'
LIGHT_INTENSITIES_FILE_NAME = 'light_intensities.txt'
MASK_FILE_NAME = 'mask.png'


class DiLiGenT(object):
    def __init__(self, path):
        self.path = path

    def load(self):
        N, m, n = self.load_normal_map()
        M, L = self.load_measurement()
        mask = self.load_mask()

        return M, L, N, m, n, mask

    def load_normal_map(self):
        n_img = cv2.imread(os.path.join(self.path, NORMAL_MAP_PNG_FILE_NAME))
        m, n, _ = n_img.shape

        N = np.loadtxt(os.path.join(self.path, NORMAL_MAP_TEXT_FILE_NAME))
        N = N.reshape(m, n, 3)
        N = N.transpose((2, 0, 1)).reshape(3, -1)

        for i in range(m * n):
            norm = np.linalg.norm(N[:, i])
            if norm != 0:
                N[:, i] /= norm

        return N, m, n

    def load_mask(self):
        mask = cv2.imread(os.path.join(self.path, MASK_FILE_NAME))[:, :, 0]
        mask = mask.reshape(-1)

        return mask

    def load_measurement(self):
        L = self.load_light_directions()
        intensities = self.load_light_intensities()
        light_num, _ = L.shape

        ######
        file_name = '{0:03d}.png'.format(1)
        m_img = cv2.imread(os.path.join(self.path, file_name))
        m, n, _ = m_img.shape

        ######
        M = np.zeros(shape=(m * n, light_num, 3), dtype=np.float32)
        for l in range(light_num):
            file_name = '{0:03d}.png'.format(l + 1)
            m_img = cv2.imread(os.path.join(self.path, file_name))[:, :, ::-1]

            m_img = m_img.reshape(-1, 3)
            M[:, l, :] = m_img

        for l in range(light_num):
            for c in range(3):
                M[:, l, c] /= intensities[l, c]

        M /= np.max(M)

        return M, L

    def load_light_directions(self):
        L = np.loadtxt(os.path.join(self.path, LIGHT_DIRECTIONS_FILE_NAME))
        return L

    def load_light_intensities(self):
        I = np.loadtxt(os.path.join(self.path, LIGHT_INTENSITIES_FILE_NAME))
        return I
