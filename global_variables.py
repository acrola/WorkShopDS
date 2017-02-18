from imports import *

# Data paths
path_complete_data = os.path.join('merged_data_ready', 'merged_data.csv')
zip_file_path = os.path.join('raw_data', 'DB_Data', 'Edstast_data.zip')
path = os.path.join('raw_data', 'DB_Data', 'Edstast_data.csv')
path_fixed = os.path.join('raw_data', 'DB_Data', 'Edstast_data_fixed.csv')
input_labels = os.path.join('raw_data', 'Labels', 'Happy_Planet_Index_Data')

# Paths for the graphical map visualization use
countries_codes = os.path.join('raw_data', 'DB_Data', 'WDI_Country.csv')
shapefile = os.path.join('map_files', 'ne_10m_admin_0_countries')
template_image = os.path.join('map_files', 'imgfile.png')
globe_plots = 'globe_plots'
uncorrolated_plots = 'uncorrolated_images'

# Dumped models path
dumped_models = 'dumped_models'

# Years with labels
rellevant_years_for_labels = ['1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', \
                              '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', \
                              '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', \
                              '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', \
                              '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2009', '2012', '2016']
rellevant_years = [year + '.00' for year in rellevant_years_for_labels]
classifiers = ['Random_Forest', 'Linear_Regression', 'Lasso', 'Ridge', 'Kernel_Ridge']

turn_on_exec = 'Run long executions'
turn_off_exec = 'Skip long executions'
turn_on_plots = 'Show plots'
turn_off_plots = 'Don\'t show plots'

# Ignore warnings
warnings.filterwarnings('ignore')
