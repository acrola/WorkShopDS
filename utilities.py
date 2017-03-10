from global_variables import *

class DataPreparation:
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
    # Unzipping the CSV File
    def unzipfile(zipped):
        zip_ref = zipfile.ZipFile(zipped, 'r')
        zip_ref.extractall(os.path.dirname(os.path.realpath(zip_file_path)))
        zip_ref.close()

    @staticmethod
    # The Main Data Extract Function
    # Run to Extract Data (invokes all the other functions above)
    def obtainDataFromLocalDBs():
        f = FloatProgress(min=0, max=100)
        display(f)
        # unzip the dataset
        DataPreparation.unzipfile(zip_file_path)
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
    def plotDataOnMap(data, year='mean', feature="Happy Planet Index", binary=False, descripton='', show_plot=True):
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
        fig = plt.figure(figsize=(20, 10))

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
        if not show_plot:
            plt.close()
        f.value += 20

    @staticmethod
    def interactMaps(overall_data,corr_features):
        def plotMap(request):
            if request == 'None':
                print('Please choose an option')
            if request == 'Plot the Happy Planet Index over the globe':
                MapVisualizations.plotDataOnMap(overall_data, feature='Happy Planet Index', year='mean')
            if request == 'Plot the 1st most correlated feature over the globe':
                MapVisualizations.plotDataOnMap(overall_data, feature=corr_features[0], year='mean')
            if request == 'Plot both for comparison':
                imgfile1 = os.path.join(globe_plots, 'Happy Planet Index' + '_' + 'mean' + '.png')
                imgfile2 = os.path.join(globe_plots, corr_features[0] + '_' + 'mean' + '.png')
                ImagesUtils.show2Images(imgfile1, imgfile2)

        interact(plotMap,\
                 request=RadioButtons(options= ['None', 'Plot the Happy Planet Index over the globe',\
                                                'Plot the 1st most correlated feature over the globe',\
                                                'Plot both for comparison'],\
                                      description='Select image to plot:', disabled=False))
class DataVisualizations:
    @staticmethod
    def twoDimPCAandClustering(factors,data_type_name):
        # Initialize the model with 2 parameters -- number of clusters and random state.
        file_data_name = data_type_name.replace(" ", "_")
        kmeans_model = KMeans(n_clusters=5, random_state=1)
        # Get only the numeric columns from games.
        # Fit the model using the good columns.
        kmeans_model.fit(factors)
        # Get the cluster assignments.
        labels = kmeans_model.labels_
        # Create a PCA model.
        pca_2 = PCA(2)
        # Fit the PCA model on the numeric columns from earlier.
        plot_columns = pca_2.fit_transform(factors)
        # Make a scatter plot of each game, shaded according to cluster assignment.
        fig1 = plt.figure(figsize=(5, 4))
        # Saving plot as an image
        ax = fig1.add_subplot(1, 1, 1)  # one row, one column, first plot
        ax.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
        ax.set_title("Two dim. PCA")
        ax.set_xlabel("Eigenvector 1")
        ax.set_ylabel("Eigenvector 2")
        plot_path = os.path.join(outliers_detection, file_data_name + '_PCA')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=.2)
        plt.close(fig1)
        return plot_columns, labels
    @staticmethod
    def allDataTwoDimPCAandClustering(request):
        if request == "None":
            print('Please choose an option')
        else:
            file_data_name = request.replace(" ", "_")
            image_initial_path = os.path.join(outliers_detection, file_data_name + '_PCA.png')
            fig1 = plt.figure(1)
            img1 = mpimg.imread(image_initial_path)
            imgplot1 = plt.imshow(img1)
            imgplot1.axes.get_xaxis().set_visible(False)
            imgplot1.axes.get_yaxis().set_visible(False)

    @staticmethod
    def simple2Dgraph(x_axis, title, xlabel, ylabel, ylim_start, ylim_end, ys, definitions, colors, save_name=''):
        for y, c, defi in zip(ys, colors, definitions):
            lines = plt.plot(x_axis.tolist(), y.tolist(), color=c, label=defi)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.ylim(ylim_start, ylim_end)
        plt.legend()

        if save_name != '':
            plot_path = os.path.join(measurements_results, save_name).replace(" ", "_")
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=.2)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def distPlot(x_axis, title, xlabel, ylabel, bins, kde):
        sns.distplot(x_axis, bins = bins, kde = kde)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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
    def show2Images(field1, field2):
        fig = plt.figure(figsize=(20,10))
        a = fig.add_subplot(1, 2, 1)
        img1 = mpimg.imread(field1)
        imgplot1 = plt.imshow(img1)
        imgplot1.axes.get_xaxis().set_visible(False)
        imgplot1.axes.get_yaxis().set_visible(False)
        img2 = mpimg.imread(field2)
        a = fig.add_subplot(1, 2, 2)
        imgplot2 = plt.imshow(img2)
        imgplot2.axes.get_xaxis().set_visible(False)
        imgplot2.axes.get_yaxis().set_visible(False)
class alternativeModles_string:
    def __init__(self,rlm_initial_r2, rlm_final_r2, rlm_n_rows_dropped, pca_r2_compared, pca_decision):
        self.rlm_initial_r2 = rlm_initial_r2
        self.rlm_final_r2 = rlm_final_r2
        self.rlm_n_rows_dropped = rlm_n_rows_dropped
        self.pca_r2_compared = pca_r2_compared
        self.pca_decision = pca_decision
class alternativeModel():
    def __init__(self,runType,train_data,train_factors,train_class,train_countries,test_data,test_factors,test_class,test_countries):
        self.train_countries = train_countries
        self.test_countries  = test_countries
        if runType == 'main data':
            self.train_data = train_data.copy()
            self.test_data = test_data.copy()
            self.train_class = train_class.copy()
            self.test_class = test_class.copy()
            self.train_factors = train_factors.copy()
            self.test_factors = test_factors.copy()
        if runType == 'no countries' or runType == 'no years':
            self.train_data  = train_data.copy()
            self.test_data   = test_data.copy()
            self.train_class = train_class.copy()
            self.test_class  = test_class.copy()
            if runType == 'no countries':
                self.train_factors = train_factors.drop(train_data.filter(regex=("country_.*")).columns, axis=1)
                self.test_factors  = test_factors.drop(test_data.filter(regex=("country_.*")).columns,axis=1)
            if runType == 'no years':
                self.train_factors = train_factors.drop(['year'],axis=1)
                self.test_factors  = test_factors.drop(['year'],axis=1)
        if runType == 'no country' or runType == 'no year':
            all_factors   = pd.concat([train_factors, test_factors])
            all_countries = pd.concat([train_countries, test_countries])
            all_classes   = pd.concat([train_class, test_class])
            all_data                       = all_factors
            all_data['country']            = all_countries
            all_data['Happy Planet Index'] = all_classes
            if runType == 'no country':
                self.train_data = all_data[all_data.country!='Israel']
                self.test_data  = all_data[all_data.country=='Israel']
                self.train_data = self.train_data.drop(['country_Israel'])
                self.test_data  = self.test_data.drop(['country_Israel'])
            if runType == 'no year':
                self.train_data = all_data[all_data.country!=4]
                self.test_data  = all_data[all_data.country==4]
            self.train_factors = self.train_data.drop(['country','Happy Planet Index'],axis=1)
            self.test_factors  = self.test_data.drop(['country','Happy Planet Index'],axis=1)
            self.train_class   = self.train_data['Happy Planet Index']
            self.test_class    = self.test_data['Happy Planet Index']
class OutliersDetection:
    @staticmethod
    def linearityProving(train_factors, train_class):
        print("Applying OLS on train data and checking model assumptions")
        regr = linear_model.LinearRegression()
        regr.fit(train_factors, train_class)
        r2 = regr.score(train_factors, train_class)
        print("train R^2: %.4f " % (r2))
        res = train_class - regr.predict(train_factors)
        y, x = res, regr.predict(train_factors)
        print("residuals appear to behave randomly, it suggests that the linear model fits the data well")
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(1, 1, 1)  # one row, one column, first plot
        ax.scatter(x, y, c="blue", alpha=.1, s=300)
        ax.set_title("residuals vs. predicted:")
        ax.set_xlabel("predicted")
        ax.set_ylabel("residuals)")
        plt.show()
        print("residuals appear to be normally distributed")
        fig = sm.qqplot(res)
        plt.show()
    @staticmethod
    def avgR2(g_factors, g_class, n_iter):
        sum = 0.0
        for i in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(g_factors, g_class, test_size=0.4, random_state=1)
            regr = linear_model.LinearRegression()
            regr.fit(X_test, y_test)
            sum += regr.score(X_test, y_test)
        return sum / (n_iter * (1.0))
    @staticmethod
    def allDataLinearityProving(request):
        if request == "None" :
            print('Please choose an option')
        else:
            OutliersDetection.linearityProving(alternativeModles[request].train_factors,\
                                               alternativeModles[request].train_class)
    @staticmethod
    def removeOutliersRlm(train_factors, train_class, train_data, i,data_type_name):
        file_data_name = data_type_name.replace(" ", "_")
        for i in range(i):
            amount = 0
            dropped_rows = np.asarray([])
            validation_r_squared = OutliersDetection.avgR2(train_factors, train_class, 100)
            alternativeModles_strings[data_type_name].rlm_initial_r2 = \
                "• Validation R^2 before outliers\' removal, %.4f " % (validation_r_squared)
            rob = sklearn.linear_model.HuberRegressor()
            X = np.asarray(train_factors)
            Y = np.asarray(train_class)
            rob.fit(X, Y)
            y_predicted = rob.predict(X)
            # plotting res vs. pred before dropping outliers
            res = [val for val in (Y - y_predicted)]
            y, x = res, rob.predict(X)
            #Saving plot as an image
            fig1 = plt.figure(figsize=(5, 4))
            ax = fig1.add_subplot(1, 1, 1)  # one row, one column, first plot
            ax.scatter(x, y, c="blue", alpha=.1, s=300)
            ax.set_title("residuals vs. predicted - initial")
            ax.set_xlabel("predicted")
            ax.set_ylabel("residuals")
            plot_path = os.path.join(outliers_detection, file_data_name + '_residuals_initial')
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=.2)
            plt.close(fig1)
            # dropping rows
            res = [abs(val) for val in (Y - y_predicted)]
            rresid = list(zip(range(train_factors.shape[0]), res))
            not_sorted = rresid
            rresid.sort(key=lambda tup: tup[1], reverse=True)
            length = len(rresid)
            sd = np.asarray([tup[1] for tup in rresid]).std()
            mean = np.asarray([tup[1] for tup in rresid]).mean()
            deleted_index = [tup[0] for tup in rresid if tup[1] > mean + 2 * sd]
            amount += len(deleted_index)
            #dropped_rows = train_factors.take(deleted_index, axis=0, convert=True, is_copy=True)
            train_factors = train_factors.drop(train_factors.index[deleted_index])
            train_class = train_class.drop(train_class.index[deleted_index])
            train_data = train_data.drop(train_data.index[deleted_index])
            alternativeModles_strings[data_type_name].rlm_n_rows_dropped = "• %d rows were dropped" % (amount)
            train_factors.reset_index(drop=True, inplace=True)
            train_class.reset_index(drop=True, inplace=True)
            train_data.reset_index(drop=True, inplace=True)
            # res vs. pred after outliers dropping
        X = np.asarray(train_factors)
        Y = np.asarray(train_class)
        validation_r_squared = OutliersDetection.avgR2(train_factors, train_class, 100)
        alternativeModles_strings[data_type_name].rlm_final_r2 = \
            "• Validation R^2 after outliers\' removal, %.4f " % (validation_r_squared)
        rob = sklearn.linear_model.HuberRegressor()
        rob.fit(X, Y)
        y_predicted = rob.predict(X)
        res = [val for val in (Y - y_predicted)]
        y, x = res, rob.predict(X)
        # Saving plot as an image
        fig2 = plt.figure(figsize=(5, 4))
        ax = fig2.add_subplot(1, 1, 1)
        ax.scatter(x, y, c="purple", alpha=.1, s=300)
        ax.set_title("residuals vs. predicted - final")
        ax.set_xlabel("predicted")
        ax.set_ylabel("residuals")
        plot_path = os.path.join(outliers_detection, file_data_name + '_residuals_final')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=.2)
        plt.close(fig2)
        return train_factors, train_class, train_data

    @staticmethod
    def allDataRemoveOutliersRlm(request):
        if request == "None":
            print('Please choose an option')
        else:
            alternativeModles[request].train_factors, alternativeModles[request].train_class, alternativeModles[
                request].train_data = \
                OutliersDetection.removeOutliersRlm(alternativeModles[request].train_factors,
                    alternativeModles[request].train_class, alternativeModles[request].train_data,\
                                                    1,request)
    @staticmethod
    def showResidualsRemoval(request):
        if request == "None":
            print('Please choose an option')
        else:
            file_data_name = request.replace(" ", "_")
            image_initial_path = os.path.join(outliers_detection, file_data_name + '_residuals_initial.png')
            image_final_path = os.path.join(outliers_detection, file_data_name + '_residuals_final.png')
            print(alternativeModles_strings[request].rlm_initial_r2 + "\n")
            print(alternativeModles_strings[request].rlm_n_rows_dropped + "\n")
            print(alternativeModles_strings[request].rlm_final_r2 + "\n")
            ImagesUtils.show2Images(image_initial_path, image_final_path)

    @staticmethod
    def printOutlierCountries(outliers_df_AltModels,outliers_indecies_AltModels):
        def printCountries(request):
            if request == "None":
                print("Please choose an option")
            else:
                print(outliers_df_AltModels[request]['country'].head(min(10, len(outliers_indecies_AltModels[request]))))
        interact(printCountries, \
                 request=RadioButtons(options=['None']+dataTypes, \
                                          description='Select which data\'s outliers countries to print:',\
                                      disabled=False))

    @staticmethod
    def removeOutliersPCA(train_factors, train_class, train_data, outliers_indecies):
        enet = ElasticNetCV(max_iter=5000, cv=10, n_jobs=-1)
        enet.fit(train_factors, train_class)
        train_r_squared_with_outliers = enet.score \
            (train_factors, train_class)
        training_data_without_outliers = train_factors.drop(outliers_indecies, inplace=False)
        training_class_without_outliers = train_class.drop(outliers_indecies, inplace=False)
        enet.fit(training_data_without_outliers, training_class_without_outliers)
        train_r_squared_without_outliers = enet.score \
            (training_data_without_outliers, training_class_without_outliers)
        print('\tR^2 on validation set with outliers:', train_r_squared_with_outliers, \
              ', and without outliers:', train_r_squared_without_outliers)
        if (abs(train_r_squared_without_outliers - train_r_squared_with_outliers) > 0.03):
            print('\tRemoving outliers from training set. \n')
            train_factors = train_factors.drop(train_factors.index[outliers_indecies])
            train_class = train_class.drop(train_class.index[outliers_indecies])
            train_data = train_data.drop(train_data.index[outliers_indecies])
        else:
            print('\tLeaving outliers in the training set, did not exceed 0.3 threshold in the difference between the R^2s. \n')
        return train_factors, train_class, train_data
class ResultsMeasurements():
    def __init__(self, loadModel, trainData, testData, trainFactors, testFactors, trainClass, testClass, model, modelName):
        # A dataframe containing Years, GDP Per Capita, Labels, Predictions
        self.model = model
        self.modelName = modelName
        self.trainRelevantData = pd.DataFrame(trainData['GDP per capita (constant 2005 US$)'])
        self.trainRelevantData['GDP'] = trainData['GDP per capita (constant 2005 US$)']
        self.trainRelevantData['year'] = trainData['year']
        self.trainRelevantData['country'] = trainData['country']
        self.trainRelevantData['label'] = pd.DataFrame(trainClass)
        model_file = modelName.replace(" ", "_")

        if not loadModel:
            self.model.fit(trainFactors, trainClass)
            ModelDump.dumpModelToFile(model_file, model)
            print("dump")
        else:
            self.model = ModelDump.loadModelFromFile(model_file)
        self.trainRelevantData['prediction'] = self.model.predict(trainFactors)
        self.trainRelevantData.is_copy = False
        self.trainFactors = trainFactors

        self.testRelevantData = pd.DataFrame(testData['GDP per capita (constant 2005 US$)'])
        self.testRelevantData['GDP'] = testData['GDP per capita (constant 2005 US$)']
        self.testRelevantData['year'] = testData['year']
        self.testRelevantData['country'] = testData['country']
        self.testRelevantData['label'] = pd.DataFrame(testClass)
        self.testRelevantData['prediction'] = self.model.predict(testFactors)
        self.testRelevantData.is_copy = False
        self.testFactors = testFactors

        self.testPlotToMap = self.testRelevantData['prediction'].to_frame()
        self.testPlotToMap['year'] = self.testRelevantData['year']
        self.testPlotToMap['country'] = self.testRelevantData['country']
        self.testPlotToMap['label'] = self.testRelevantData['label']

        self.RSquaredTrain = self.model.score(self.trainFactors, self.trainRelevantData['label'])
        self.RSquaredTest = self.model.score(self.testFactors, self.testRelevantData['label'])
        self.train_mean_label = self.trainRelevantData['label'].mean()
        self.train_mean_prediction = self.trainRelevantData['prediction'].mean()
        self.test_mean_label = self.testRelevantData['label'].mean()
        self.test_mean_prediction = self.testRelevantData['prediction'].mean()
        self.error_percentage_train = self.errPercentageCalc(self.trainRelevantData['prediction'],
                                                      self.trainRelevantData['label'])
        self.error_percentage_test = self.errPercentageCalc(self.testRelevantData['prediction'],
                                                     self.testRelevantData['label'])
        self.comparison_parameters_df = pd.DataFrame([self.RSquaredTrain,self.RSquaredTest,self.train_mean_label, \
                                      self.train_mean_prediction,self.test_mean_label,self.test_mean_prediction, \
                                      self.error_percentage_train,self.error_percentage_test])

    def RsquaredGraph(self, r2_train, r2_test, x_axis):
        DataVisualizations.simple2Dgraph(r2_train[0],
                                         self.modelName + '\n R^2 per ' + x_axis + ', Train (blue) vs. Test(green)',
                                         x_axis,
                                         'R^2', -4, 1, \
                                         [r2_train[1], r2_test[1]], ['R^2 Train', 'R^2 Test'], ['b', 'g'])

    def RsquaredSeriesYear(self, data, x_axis):
        RsquaredSeries = pd.DataFrame([[i, r2_score(data[data[x_axis] == i].label, data[data[x_axis] == i].prediction)] \
                                       for i in data[x_axis].unique()])
        return RsquaredSeries.sort_values(by=0, ascending=1)

    def RsquaredSeriesGDP(self, data, x_axis):
        sortedData = data.sort_values(by='GDP', ascending=1)
        sortedData = np.array_split(sortedData, 30)
        RsquaredSeries = pd.DataFrame([[sortedData[i].iloc[[0]]['GDP'].item(), \
                                        r2_score(sortedData[i].label, sortedData[i].prediction)] for i in
                                       range(len(sortedData))])
        return RsquaredSeries.sort_values(by=0, ascending=1)

    def RSquaredResults(self):
        print("R^2 for Train data = " + str(self.RSquaredTrain))
        print("R^2 for Test data = " + str(self.RSquaredTest))

        # RSquaredTrainYears = self.RsquaredSeriesYear(self.trainRelevantData, 'year')
        # RSquaredTestYears = self.RsquaredSeriesYear(self.testRelevantData, 'year')
        #
        # RSquaredTrainGDPs = self.RsquaredSeriesGDP(self.trainRelevantData, 'GDP')
        # RSquaredTestGDPs = self.RsquaredSeriesGDP(self.testRelevantData, 'GDP')
        #
        # # R^2 per year, per GDP
        # self.RsquaredGraph(RSquaredTrainYears, RSquaredTestYears, 'Year')
        # self.RsquaredGraph(RSquaredTrainGDPs, RSquaredTestGDPs, 'GDP')

    def DistributionNumericCalc(self, predictions):
        return stats.kstest(predictions, 'norm')

    def DistributionGraphicCalc(self, predictions, binsNum, title):
        sns.distplot(predictions, bins=binsNum, kde=True)
        plt.title(self.modelName + '\n Histogram of Happy Planet Index values: ' + title)
        plt.xlabel('HPI')
        plt.ylabel('density')

        file_name = (self.modelName + ' Histogram ' + title).replace(" ", "_")

        plot_path = os.path.join(measurements_results, file_name)
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=.2)
        plt.close()
    def DistributionResults(self):
        # print("KSTEST results, train: " + str(self.DistributionNumericCalc(self.trainRelevantData['prediction'])))
        # print("KSTEST results, test : " + str(self.DistributionNumericCalc(self.testRelevantData['prediction'])))

        # self.DistributionGraphicCalc(self.trainRelevantData['label'], 30, "train label")
        # self.DistributionGraphicCalc(self.trainRelevantData['prediction'], 30, "train prediction")

        label_title = "test label"
        prediction_title = "test prediction"
        self.DistributionGraphicCalc(self.testRelevantData['label'], 30, label_title)
        self.DistributionGraphicCalc(self.testRelevantData['prediction'], 30, prediction_title)
        ImagesUtils.show2Images(os.path.join(measurements_results, (self.modelName + ' Histogram ' + label_title+".png").replace(" ", "_"))\
                                ,os.path.join(measurements_results, (self.modelName + ' Histogram ' + prediction_title+".png").replace(" ", "_")))


    def MeanPredictionGraph(self, prediction_train, prediction_test, x_axis):
        DataVisualizations.simple2Dgraph(prediction_test[0],
                                         self.modelName + '\n HPI per ' + x_axis + ', Test Prediction mean (blue) vs. Test Label mean (green)',
                                         x_axis, 'Prediction', 0, 100, \
                                         [prediction_test[1], prediction_test[2]], ['Test Prediction', 'Test Class'],
                                         ['b', 'g'],save_name=self.modelName+' Prediction mean '+x_axis)

    def MeanPredictionSeriesYear(self, data, x_axis):
        MeanPredictionSeries = pd.DataFrame(
            [[i, data[data[x_axis] == i].prediction.mean(), data[data[x_axis] == i].label.mean()] \
             for i in data[x_axis].unique()])
        return MeanPredictionSeries.sort_values(by=0, ascending=1)

    def MeanPredictionSeriesGDP(self, data, x_axis):
        sortedData = data.sort_values(by='GDP', ascending=1)
        sortedData = np.array_split(sortedData, 30)
        MeanPredictionSeries = pd.DataFrame([[sortedData[i].iloc[[0]]['GDP'].item(), \
                                              sortedData[i].prediction.mean(), sortedData[i].label.mean()] for i in
                                             range(len(sortedData))])
        return MeanPredictionSeries.sort_values(by=0, ascending=1)

    def MeanPredictionResults(self):
        print("The mean HPI of the train data: " + str(self.train_mean_label))
        print("The mean prediction of the train data: " + str(self.train_mean_prediction))
        print("The mean HPI of the test data : " + str(self.test_mean_label))
        print("The mean prediction of the test data : " + str(self.test_mean_prediction))

        MeanPredictionTrainYears = self.MeanPredictionSeriesYear(self.trainRelevantData, 'year')
        MeanPredictionTestYears = self.MeanPredictionSeriesYear(self.testRelevantData, 'year')

        MeanPredictionTrainGDPs = self.MeanPredictionSeriesGDP(self.trainRelevantData, 'GDP')
        MeanPredictionTestGDPs = self.MeanPredictionSeriesGDP(self.testRelevantData, 'GDP')

        # Mean Prediction per year, per GDP
        self.MeanPredictionGraph(MeanPredictionTrainYears, MeanPredictionTestYears, 'Year')
        self.MeanPredictionGraph(MeanPredictionTrainGDPs, MeanPredictionTestGDPs, 'GDP')
        ImagesUtils.show2Images(
            os.path.join(measurements_results, (self.modelName+' Prediction mean Year.png').replace(" ", "_")) \
            , os.path.join(measurements_results,
                           (self.modelName+' Prediction mean GDP.png').replace(" ", "_")))

    def errPercentage(self, label, prediction):
        return (abs(label - prediction) / label) * 100

    def errPercentageCalc(self, label, prediction):
        errTable = pd.DataFrame({'label': label, 'prediction': prediction})
        errTable['errPercentage'] = errTable.apply(lambda row: self.errPercentage(row['label'], row['prediction']),
                                                   axis=1)
        return errTable['errPercentage'].mean()

    def ErrorPercentageGraph(self, errPer_train, errPer_test, x_axis):
        DataVisualizations.simple2Dgraph(errPer_train[0],
                                         self.modelName + '\n Error Percentage per ' + x_axis + ', Train vs. Test',
                                         x_axis, 'Error Percentage', 0, 100, \
                                         [errPer_train[1], errPer_test[1]],
                                         ['Error Percentage Train', 'Error Percentage Test'], ['b', 'g'], \
                                         save_name=self.modelName + ' Error Percentage ' + x_axis)

    def ErrorPercentageSeriesYear(self, data, x_axis):
        ErrorPercentage = pd.DataFrame(
            [[i, self.errPercentageCalc(data[data[x_axis] == i].label, data[data[x_axis] == i].prediction)] \
             for i in data[x_axis].unique()])
        return ErrorPercentage.sort_values(by=0, ascending=1)

    def ErrorPercentageSeriesGDP(self, data, x_axis):
        sortedData = data.sort_values(by='GDP', ascending=1)
        sortedData = np.array_split(sortedData, 30)
        ErrorPercentage = pd.DataFrame([[sortedData[i].iloc[[0]]['GDP'].item(), \
                                         self.errPercentageCalc(sortedData[i].label, sortedData[i].prediction)] for i in
                                        range(len(sortedData))])
        return ErrorPercentage.sort_values(by=0, ascending=1)

    def ErrorPercentageResults(self):
        print("Error Percentage for Train data = " + str(self.error_percentage_train))
        print("Error Percentage for Test data = " + str(self.error_percentage_test))

        ErrorPercentageTrainYears = self.ErrorPercentageSeriesYear(self.trainRelevantData, 'year')
        ErrorPercentageTestYears = self.ErrorPercentageSeriesYear(self.testRelevantData, 'year')

        ErrorPercentageTrainGDPs = self.ErrorPercentageSeriesGDP(self.trainRelevantData, 'GDP')
        ErrorPercentageTestGDPs = self.ErrorPercentageSeriesGDP(self.testRelevantData, 'GDP')

        # R^2 per year, per GDP
        self.ErrorPercentageGraph(ErrorPercentageTrainYears, ErrorPercentageTestYears, 'Year')
        self.ErrorPercentageGraph(ErrorPercentageTrainGDPs, ErrorPercentageTestGDPs, 'GDP')

        ImagesUtils.show2Images(
            os.path.join(measurements_results, (self.modelName+' Error Percentage Year.png').replace(" ", "_")) \
            , os.path.join(measurements_results,
                           (self.modelName+' Error Percentage GDP.png').replace(" ", "_")))

    def labelPredictionComparisonPlot(self):
        MapVisualizations.plotDataOnMap(self.testPlotToMap, year='mean', feature='label', binary=False, \
                                        descripton='Happy Planet Index label on test dataset',show_plot=False)
        MapVisualizations.plotDataOnMap(self.testPlotToMap, year='mean', feature='prediction', binary=False, \
                                        descripton='Happy Planet Index prediction on test dataset', show_plot=False)
        ImagesUtils.show2Images(os.path.join(globe_plots, 'label_mean.png'), os.path.join(globe_plots,'prediction_mean.png'))
    def plotForModel(self, request):
        if request == 'None':
            print("Please choose an option from the bar above")
        if request == 'Error Percentage Results':
            self.ErrorPercentageResults()
        if request == 'Mean Prediction Results':
            self.MeanPredictionResults()
        if request == 'R-Squared Results':
            self.RSquaredResults()
        if request == 'Distribution Results':
            self.DistributionResults()
    def interactResults(self):
        requests = widgets.ToggleButtons(
                    options=['None', 'R-Squared Results', 'Distribution Results', \
                                         'Mean Prediction Results', 'Error Percentage Results'],
                    description='Results:',
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip='Description',
                #     icon='check'
                )
        plotButtons = widgets.interactive(self.plotForModel, request=requests)
        return plotButtons
    @staticmethod
    def tabDisplay(models):
        children = [models[data].interactResults() for data in dataTypes]
        tab = widgets.Tab(children=children)
        for i in range(len(dataTypes)):
            tab.set_title(i, dataTypes[i])
        return tab

class ModelDump():
    @staticmethod
    def dumpModelToFile(name, model):
        with open(os.path.join(dumped_models, name + '.pkl'), 'wb') as fid:
            cPickle.dump(model, fid)

    @staticmethod
    def loadModelFromFile(name):
        with open(os.path.join(dumped_models, name + '.pkl'), 'rb') as fid:
            model = cPickle.load(fid)
        return model
class AlternativeModels():
    @staticmethod
    def creatAlternativeModels(train_data, train_factors, train_class, train_countries, test_data,\
                               test_factors, test_class, test_countries):
        global alternativeModles
        alternativeModles = [alternativeModel(runType, train_data, train_factors, train_class, train_countries, test_data, \
                                              test_factors, test_class, test_countries) for runType in dataTypes]
        alternativeModles = dict([(dataTypes[i], alternativeModles[i]) for i in range(len(dataTypes))])
        global alternativeModles_strings
        alternativeModles_strings = dict([(data, alternativeModles_string("", "", "", "", "")) for data in dataTypes])
        return alternativeModles

    @staticmethod
    def updateAlternativeModles(newAlternativeModles):
        global alternativeModles
        alternativeModles = newAlternativeModles

    @staticmethod
    def printAlters():
        for data in dataTypes:
            print(data)
            print(alternativeModles[data].train_factors.shape)
            print(alternativeModles[data].test_factors.shape)
def clicked(b):
    temp = b.tooltip
    if b.value == True:
        b.tooltip = b.description
        b.description = temp
        b.value = False
    else:
        b.tooltip = b.description
        b.description = temp
        b.value = True

button_exec = widgets.Button(description=turn_on_exec, tooltip=turn_off_exec, value=True)
button_exec.on_click(clicked)

button_plots = widgets.Button(description=turn_off_plots, tooltip=turn_on_plots, value=True)
button_plots.on_click(clicked)

display(button_exec)
display(button_plots)
