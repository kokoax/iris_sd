#! coding: utf-8
import sys
import random
import math
import numpy as np
# import matplotlib.pyplot as plt
import re
from sklearn import datasets

class Iris_SD:
    def __init__(self):
        self.nrow = 0
        self.ncol = 0
        self.data_flg = 0
        self.all_data_sets = None
        self.data_sets = self.getDataSets()
        # print(self.get_avg_from_column(0))
        avg = self.get_avg_from_column(0)
        dis = self.get_dispersion_from_column(0, avg)
        sd  = math.sqrt(dis)
        print("avg: ", avg)
        print("dis: ", dis)
        print("sd : ", sd )

    def get_avg_from_column(self, column):
        _sum = 0
        for i in range(self.nrow):
            _sum += self.data_sets[i]['data'][column]
        return _sum/self.nrow

    def get_dispersion_from_column(self, column, avg):
        _dis = 0
        for i in range(self.nrow):
            _dis += (self.data_sets[i]['data'][column]-avg) ** 2
        return _dis/self.nrow

    def getDataSets(self):
        if self.data_flg == 0:
            data_sets = datasets.load_iris()
        elif self.data_flg == 1:
            data_sets = datasets.load_digits()

        self.all_data_sets = data_sets
        self.nrow, self.ncol = data_sets.data.shape
        return [
                {
                    'name':data_sets.target[i],
                    'data':data_sets.data[i],
                    'cluster':-1
                } for i in range(self.nrow)
        ]

isd = Iris_SD()

