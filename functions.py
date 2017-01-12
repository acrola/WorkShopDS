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
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.model_selection import train_test_split

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

# for images comparison:
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# model for feature selection:
from sklearn import datasets, linear_model, decomposition
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as sp
import sklearn.feature_selection as fs
from sklearn import kernel_ridge
#import skfeature as skf

# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.grid_search import GridSearchCV

# Imports for kernel ridge:
from sklearn.model_selection import GridSearchCV


# loading bar
from ipywidgets import FloatProgress
from IPython.display import display

# Data paths
path_complete_data = os.path.join('merged_data_ready','merged_data.csv')
path = os.path.join('raw_data','DB_Data','Edstast_data.csv')
path_fixed = os.path.join('raw_data','DB_Data','Edstast_data_fixed.csv')
input_labels = os.path.join('raw_data','Labels','Happy_Planet_Index_Data')

# Paths for the graphical map visualization use
countries_codes = os.path.join('raw_data','DB_Data','WDI_Country.csv')
shapefile = os.path.join('map_files', 'ne_10m_admin_0_countries')
template_image = os.path.join('map_files', 'imgfile.png')
globe_plots = 'globe_plots'
uncorrolated_plots = 'uncorrolated_images'

# Years with labels
rellevant_years_for_labels = ['1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',\
                              '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',\
                              '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979',\
                              '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989',\
                              '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2009', '2012', '2016']
rellevant_years = [year + '.00' for year in rellevant_years_for_labels]

class DataPreparation():
    @staticmethod
    def retriveMergedFilePath():
        return path_complete_data

    @staticmethod
    # Merge the data with the labels
    def mergeDataWithLabels(working_frame, labels):
        result = pd.merge(working_frame, labels, how='inner', on=['country', 'year'])
        result.to_csv(path_complete_data)

    @staticmethod
    # Cleaning the CSV Files Out From Commas
    def cleanCommasFromCSV():
        with open(path, "r", newline="") as infile, open(path_fixed, "w", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow(item.replace(",", "") for item in row)

    @staticmethod
    # Obtain The Labeled Data
    def getDataFrameForLabelCSV(path, year):
        df = pd.read_csv(path, skiprows=0, usecols=[1, 8])
        df.loc[:, 'year'] = pd.Series(float(rellevant_years[rellevant_years_for_labels.index(year)]), index=df.index)
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        df.drop(df.index[[0]])
        return df

    @staticmethod
    # Take Data From DataSets
    def extractDataFromCSV():
        df = pd.read_csv(path, header=None, skiprows=0, encoding='iso-8859-1')
        df.drop(df.columns[[1, 3]], axis=1, inplace=True)
        df.loc[0, 0] = 'country'
        df.columns = df.loc[0]
        df = pd.pivot_table(df, index='country', columns='Indicator Name')
        df = df.stack(level=0)
        df.reset_index(inplace=True)
        df.rename(columns={0: 'year'}, inplace=True)
        df.rename(columns={"Indicator Name": 'series'}, inplace=True)
        df.to_csv(path_complete_data, encoding='iso-8859-1')
        df = pd.read_csv(path_complete_data, encoding='iso-8859-1')
        df.drop(df.columns[[0]], axis=1, inplace=True)
        return df

    @staticmethod
    # The Main Data Extract Function
    # Run to Extract Data (invokes all the other functions above)
    def obtainDataFromLocalDBs():
        f = FloatProgress(min=0, max=100)
        display(f)

        # extract the labels dataframe from the csv files
        lis = []
        for year in rellevant_years_for_labels:
            path = os.path.join(input_labels + '_' + year + '.csv')
            df = DataPreparation.getDataFrameForLabelCSV(path, year)
            lis.append(df)
        labels_df = pd.concat(lis)
        f.value += 10
        # extract all the data dataframe from the csv files
        DataPreparation.cleanCommasFromCSV()
        f.value += 20
        df = DataPreparation.extractDataFromCSV()
        f.value += 20

        # merge (by inner join) the data with the labels
        DataPreparation.mergeDataWithLabels(df, labels_df)
        f.value += 50

class MapVisualizations:
    @staticmethod
    def plotDataOnMap(data, year='mean', feature="Happy Planet Index", binary=False, descripton=''):
        if binary:
            num_colors = 2
        else:
            num_colors = 9
        cols = ['country', feature]
        splitted = feature.split()
        title = feature + ' rate per country'
        imgfile = os.path.join(globe_plots, feature + '_' + year + '.png')
        if descripton == '':
            descripton = '''
            Expected values of the {} rate of countriers. Countries without data are shown in grey.
            Data: World Bank - worldbank.org • Lables: HappyPlanetIndex - happyplanetindex.org'''.format(feature)

        gc = GeonamesCache()
        iso3_codes = list(gc.get_dataset_by_key(gc.get_countries(), 'iso3').keys())
        df = pd.read_csv(countries_codes, skiprows=0, usecols=[0, 1], encoding='iso-8859-1')
        data_map = pd.merge(df, data, how='inner', on=['country'])
        if not binary:
            if year == 'mean':
                data_map = data_map[['Country Code', 'country', feature]]
                data_map = data_map.groupby(['Country Code'], sort=False).mean()
            else:
                data_map = data_map[['Country Code', 'year', 'country', feature]]
                data_map = data_map.loc[data_map['year'] == float(year)]
                data_map = data_map[['Country Code', 'country', feature]]
                data_map = data_map.groupby(['Country Code'], sort=False).first()
        data_map.reset_index(inplace=True)
        values = data_map[feature]
        data_map.set_index('Country Code', inplace=True)
        if not binary:
            cm = plt.get_cmap('Greens')
            scheme = [cm(i / num_colors) for i in range(num_colors)]
        else:
            cm = plt.get_cmap('prism')
            scheme = [cm(i * 20 / num_colors) for i in range(num_colors)]
        bins = np.linspace(values.min(), values.max(), num_colors)
        data_map['bin'] = np.digitize(values, bins) - 1
        data_map.sort_values('bin', ascending=False).head(10)
        fig = plt.figure(figsize=(22, 12))

        ax = fig.add_subplot(111, axisbg='w', frame_on=False)
        if not binary:
            if year == 'mean':
                fig.suptitle('mean {} rate for all data'.format(' '.join(splitted[:7]), year), fontsize=30, y=.95)
            else:
                fig.suptitle('{} rate in year {}'.format(' '.join(splitted[:7]), year), fontsize=30, y=.95)
        else:
            fig.suptitle('{} rate'.format(' '.join(splitted[:7]), year), fontsize=30, y=.95)

        m = Basemap(lon_0=0, projection='robin')
        m.drawmapboundary(color='w')

        f = FloatProgress(min=0, max=100)
        display(f)

        m.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
        for info, shape in zip(m.units_info, m.units):
            iso3 = info['ADM0_A3']
            if iso3 not in data_map.index:
                color = '#dddddd'
            else:
                ind = data_map.ix[iso3, 'bin'].astype(np.int64)
                color = scheme[ind]

            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches)
            pc.set_facecolor(color)
            ax.add_collection(pc)
            f.value += 75 / len(m.units_info)

        # Cover up Antarctica so legend can be placed over it.
        ax.axhspan(0, 1000 * 1800, facecolor='w', edgecolor='w', zorder=2)

        # Draw color legend.
        ax_legend = fig.add_axes([0.35, 0.14, 0.3, 0.03], zorder=3)
        cmap = mpl.colors.ListedColormap(scheme)
        if binary:
            grads = np.linspace(0., 10)
            cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, boundaries=grads, ticks=[0, 10],
                                           orientation='horizontal')
            cb.ax.set_xticklabels(['negative', 'positive'])
        else:
            cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, boundaries=bins, ticks=bins, orientation='horizontal')
            cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])
        f.value += 5

        # Set the map footer.
        plt.annotate(descripton, xy=(-.8, -3.2), size=14, xycoords='axes fraction')
        plt.savefig(imgfile, bbox_inches='tight', pad_inches=.2)
        plt.plot()
        f.value += 20

    @staticmethod
    def plotUncorrolatedCountries(im1, im2, output):
        img1 = cv2.imread(im1, 1)
        img2 = cv2.imread(im2, 1)
        null_img = cv2.imread(template_image, 1)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        null_img = cv2.cvtColor(null_img, cv2.COLOR_BGR2GRAY)

        height1, width1 = img1.shape
        height2, width2 = img2.shape
        height3, width3 = null_img.shape

        min_h = min(height1, height2, height3)
        min_w = min(width1, width2, width3)

        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        null_img = null_img[:min_h, :min_w]

        crop_img = cv2.subtract(img1, img2)[65:900, :]

        null_img = null_img[65:900, :]
        thresh = (255 - crop_img)

        cv2.addWeighted(thresh, 0.5, null_img, 0.5, 0, thresh)
        (threshold, thresh) = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        flag = cv2.imwrite(output, thresh)
        plt.axis('off')
        plt.imshow(thresh, cmap='gray', interpolation='bicubic'), plt.show()

class DataVisualizations:
    @staticmethod
    def twoDimPCAandClustering(factors):
        # Initialize the model with 2 parameters -- number of clusters and random state.
        kmeans_model = KMeans(n_clusters=5, random_state=1)
        # Get only the numeric columns from games.
        # Fit the model using the good columns.
        kmeans_model.fit(factors)
        # Get the cluster assignments.
        labels = kmeans_model.labels_
        # Import the PCA model.

        # Create a PCA model.
        pca_2 = PCA(2)
        # Fit the PCA model on the numeric columns from earlier.
        plot_columns = pca_2.fit_transform(factors)
        # Make a scatter plot of each game, shaded according to cluster assignment.
        plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
        # Show the plot.
        plt.show()
        return plot_columns, labels
    @staticmethod
    def simple2Dgraph(x_axis,title, xlabel, ylabel, ylim_start, ylim_end, ys, colors):
        for y, c in zip(ys, colors):
            lines = plt.plot(x_axis.tolist(), y.tolist(), color=c)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.ylim(ylim_start, ylim_end)
        plt.show()

class ImagesUtils:
        @staticmethod
        def concat_images(imga, imgb):
            """
            Combines two color image ndarrays side-by-side.
            """
            ha, wa = imga.shape[:2]
            hb, wb = imgb.shape[:2]
            max_height = np.max([ha, hb])
            total_width = wa + wb
            new_img = np.zeros(shape=(max_height, total_width), dtype=np.uint8)
            new_img[:ha, :wa] = imga
            new_img[:hb, wa:wa + wb] = imgb
            return new_img

        @staticmethod
        def concat_n_images(image_path_list):
            """
            Combines N color images from a list of image paths.
            """
            output = None
            for i, img_path in enumerate(image_path_list):
                img = plt.imread(img_path)[:, :]
                if i == 0:
                    output = img
                else:
                    output = ImagesUtils.concat_images(output, img)
            return output