import pandas as pd
from matplotlib import pyplot as plt
from multi_factor import m_factor

class trade():
    def __init__(self,origin=1000000,cost=0.002):
        self.cash = origin
        self.cost = 0.002

    def pool(self,func):
        df = func()
