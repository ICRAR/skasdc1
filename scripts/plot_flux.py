import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def comparison_plot(fn):
    data = pd.read_csv(fn)