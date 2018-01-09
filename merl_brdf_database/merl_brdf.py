#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Hiroaki Santo

import numpy as np
import os

import BRDFRead


class BrdfModel(object):
    def __init__(self, model_path):
        assert os.path.exists(model_path)
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.model_name, _ = os.path.splitext(self.model_name)
        self.brdf_mat = BRDFRead.read_brdf(model_path)

        assert self.brdf_mat.shape[0] > 0

    def lookup_brdf_by_angle(self, theta_in, fi_in, theta_out, fi_out):
        """

        :param theta_in:
        :param fi_in:
        :param theta_out:
        :param fi_out:
        :return: (red_val, green_val, blue_val)
        """
        return BRDFRead.lookup_brdf_val(self.brdf_mat, theta_in, fi_in, theta_out, fi_out)

    def lookup_brdf_by_vectors(self, in_vec, out_vec):

        if in_vec[2] < 0 or out_vec[2] < 0:
            return np.array([0., 0., 0.])

        theta_in, fi_in = self.vector2polar(in_vec)
        theta_out, fi_out = self.vector2polar(out_vec)
        return self.lookup_brdf_by_angle(theta_in, fi_in, theta_out, fi_out)

    def __call__(self, light, view, normal):
        view = np.array(view).reshape(-1)
        normal = np.array(normal).reshape(-1)
        light = np.array(light).reshape(-1)

        theta_in = np.arccos(np.dot(normal, light) / (np.linalg.norm(normal) * np.linalg.norm(light)))
        theta_out = np.arccos(np.dot(normal, view) / (np.linalg.norm(normal) * np.linalg.norm(view)))

        if theta_in > np.pi / 2 or theta_out > np.pi / 2:
            return np.array((0., 0., 0.))

        foot_light = self.calc_foot_point_on_plane(normal, light)
        foot_view = self.calc_foot_point_on_plane(normal, view)

        phi_in = 0.0
        phi_out = np.arccos(np.dot(foot_light, foot_view) / (np.linalg.norm(foot_light) * np.linalg.norm(foot_view)))

        return self.lookup_brdf_by_angle(theta_in, phi_in, theta_out, phi_out)

    @staticmethod
    def vector2polar(vec):
        """

        :param np.ndarray vec: (3, )
        :return: (theta, phi)
        """
        assert np.linalg.norm(vec) > 0

        theta = np.arccos(vec[2] / np.linalg.norm(vec))
        if np.linalg.norm([vec[0], vec[1]]) == 0:
            phi = 0.0
        else:
            phi = np.sign(vec[1]) * np.arccos(vec[0] / np.linalg.norm([vec[0], vec[1]]))

        return theta, phi

    @staticmethod
    def polar2vector(theta, phi, r=1):
        """

        :param float theta: [rad]
        :param float phi: [rad]
        :param float r: default r=1
        :return: x, y, z
        """

        assert r >= 0
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return np.array((x, y, z))

    @staticmethod
    def calc_foot_point_on_plane(n, v, h=0):
        """

        :param np.ndarray n: normal vector of plane
        :param np.ndarray v:
        :param float h: plane height
        :return:
        :rtype: np.ndarray
        """
        n = np.array(n).flatten() / np.linalg.norm(n)
        v = np.array(v).flatten() / np.linalg.norm(v)
        p = v - (np.dot(n, v) - h) * n
        return p
