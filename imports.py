import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.cm
import pathlib
import csv
import os
import zipfile
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.model_selection import train_test_split
import warnings


# for map graphical view:
import matplotlib.cm
import matplotlib as mpl
from geonamescache import GeonamesCache
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap

# for PCA:
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for outliers detections
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
import statsmodels.api as sm

# for images comparison:
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

# model for feature selection:
from sklearn import datasets, linear_model, decomposition
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as sp
import sklearn.feature_selection as fs
from sklearn import kernel_ridge
# import skfeature as skf

# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# from sklearn.grid_search import GridSearchCV

# Imports for kernel ridge:
from sklearn.model_selection import GridSearchCV

# Imports for Results check
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

# cPickle for model fit download to file
import _pickle as cPickle

# loading bar
from ipywidgets import FloatProgress
from IPython.display import display
from ipywidgets import interact, interactive, RadioButtons
import ipywidgets as widgets