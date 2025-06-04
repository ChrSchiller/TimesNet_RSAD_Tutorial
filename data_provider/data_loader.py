###imports 
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

class RsTsLoader(Dataset):
    def __init__(self, root_path, win_size, flag="train", max_index=None, fixed_norm=0, 
                 use_qai=0, info="", indices_bands=[], norm_mean=None, norm_std=None):
        self.flag = flag
        self.win_size = win_size # == seq_len
        self.root_dir = root_path
        self.max_index = max_index
        self.fixed_norm = fixed_norm
        self.use_qai = use_qai
        self.info = info
        self.indices_bands = indices_bands

        ### load the dataset depending on flag value ("train", "val", "test")
        if self.flag == "train":
            self.input_files = pd.read_csv(os.path.join(root_path, 'split_data', 'train_' + str(info) + '.csv'))
        elif self.flag == "val":
            self.input_files = pd.read_csv(os.path.join(root_path, 'split_data', 'val_' + str(info) + '.csv'))
        elif self.flag == "test":
            self.input_files = pd.read_csv(os.path.join(root_path, 'split_data', 'test_' + str(info) + '.csv'))
        elif self.flag == "predict":
            self.input_files = pd.read_csv(os.path.join(root_path, 'split_data', 'test_' + str(info) + '.csv'))
        print(self.flag, " dataset length: ", len(self.input_files))

        if self.fixed_norm == 1:
            ### check if norm_mean and norm_std are empty lists
            if norm_mean:
                ### split the provided values into mean and std
                self.overall_mean = np.array(norm_mean)
                self.overall_std = np.array(norm_std)
                print("Using provided normalization values...")
            else:
            ### load the normalization parameters using root_path + split_data + train_overall_mean.npy and train_overall_std.npy
                mean_df = pd.read_csv(root_path + '/split_data/train_overall_mean_indices_bands_' + str(info) + '.csv')
                std_df = pd.read_csv(root_path + '/split_data/train_overall_std_indices_bands_' + str(info) + '.csv')
                self.overall_mean = mean_df['mean'].values
                self.overall_std = std_df['std'].values
                print("Using fixed normalization parameters...")

    def __len__(self):
        return len(self.input_files)  # number of samples in the dataset

    ### method that reads, prepares and outputs a single sample time series (here: csv file)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ### we added this while loop to avoid code crashes
        ### due to empty dataframes after quality and index criteria
        ### this case does not happen in the dataset used for the paper
        while True: 
            ### read the csv file name from the input_files dataframe
            X_name = os.path.join(self.root_dir,
                                self.input_files.iloc[idx, 0] +
                                '.csv')
            # col_list specifies the dataframe columns to use as model input
            bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2']
            collist = bands.copy()
            if self.use_qai == 1:
                collist.append('QAI')
            collist.append('DOY')

            ### read the csv file
            X = pd.read_csv(X_name, sep=',', usecols=collist)
            ### sort columns by collist
            X = X[collist]
            ### rename columns: bands + ["DOY"]
            if self.use_qai == 1:
                X.columns = bands + ["QAI"] + ["DOY"]
            else:
                X.columns = bands + ["DOY"]

            ### use QAI column to filter good quality observations
            if self.use_qai == 1:
                ### convert QAI column (integer) to 15-bit binary
                X['QAI_binary'] = X['QAI'].apply(lambda x: format(x, '015b'))
                ### decipher the binary values according to the following scheme: 
                ### https://force-eo.readthedocs.io/en/latest/howto/qai.html
                ### separate this 15-bit binary number as follows
                ### from right to left: 
                ### first bit should be stored in the field "valid"
                ### second and third bit together stored in "cloud"
                ### fourth bit stored in cloud_shadow
                ### fifth bit stored in snow
                ### sixth bit stored in water
                ### seventh and eigth bit together stored in aerosol
                ### ninth bit stored in subzero
                ### tenth bit stored in saturation
                ### eleventh bit stored in high_sun_zenith
                ### twelfth and thirteenth bit together stored in illumination
                ### fourteenth bit stored in slope
                ### fifteenth bit stored in water_vapor
                X['valid'] = X['QAI_binary'].apply(lambda x: x[14])
                X['cloud'] = X['QAI_binary'].apply(lambda x: x[12:14])
                X['cloud_shadow'] = X['QAI_binary'].apply(lambda x: x[11])
                X['snow'] = X['QAI_binary'].apply(lambda x: x[10])
                X['water'] = X['QAI_binary'].apply(lambda x: x[9])
                X['aerosol'] = X['QAI_binary'].apply(lambda x: x[7:9])
                X['subzero'] = X['QAI_binary'].apply(lambda x: x[6])
                X['saturation'] = X['QAI_binary'].apply(lambda x: x[5])
                X['high_sun_zenith'] = X['QAI_binary'].apply(lambda x: x[4])
                X['illumination'] = X['QAI_binary'].apply(lambda x: x[2:4])
                X['slope'] = X['QAI_binary'].apply(lambda x: x[1])
                X['water_vapor'] = X['QAI_binary'].apply(lambda x: x[0])
                ### remove rows with the following features: 
                ### valid == 1, cloud == 10 or 11, cloud_shadow == 1, snow == 1, water == 1, aerosol == 10 or 11, 
                ### subzero == 1, saturation == 1, high_sun_zenith == 1, illumination == 10 or 11
                X = X[(X['valid'] == "0") & 
                    (X['cloud'] != "10") &
                    (X['cloud'] != "11") & 
                    (X['cloud_shadow'] == "0") & 
                    (X['snow'] == "0") & 
                    (X['water'] == "0") & 
                    (X['aerosol'] != "10") & # 'high (aerosol optical depth > 0.6, use with caution)' -> could be used
                    (X['aerosol'] != "11") & 
                    (X['subzero'] == "0") & 
                    (X['saturation'] == "0") & 
                    (X['high_sun_zenith'] == "0") & 
                    (X['illumination'] != "10") & # 'poor (incidence angle > 90°, low quality for topographic correction)' -> could be used
                    (X['illumination'] != "11") 
                ]
                X = X.drop(columns=['QAI', 'QAI_binary', 'valid', 'cloud', 'cloud_shadow', 'snow', 'water', 'aerosol', 'subzero', 'saturation', 'high_sun_zenith', 'illumination', 'slope', 'water_vapor'])

            ### list of indices to check
            indices_to_check = ["DSWI", "NDWI", "CLRE", "NDREI2", "NDREI1", "SWIRI", "CRSWIR", "NGRDI", "SRSWIR", "LWCI"]
            if any(index in self.indices_bands for index in indices_to_check):
                # Compute the indices
                if "DSWI" in self.indices_bands:
                    ### Galvao et al 2005, 
                    ### healthy between 0 and 1
                    ### stressed > 1
                    X['DSWI'] = (X['BNR'] + X['GRN']) / (X['SW1'] + X['RED'])
                if "NDWI" in self.indices_bands:
                    ### McFeeters 1996
                    ### usually -1 until 1
                    X['NDWI'] = (X['GRN'] - X['BNR']) / (X['GRN'] + X['BNR'])
                if "CLRE" in self.indices_bands:
                    X['CLRE'] = (X['RE3'] / X['RE1']) - 1
                if "NDREI2" in self.indices_bands:
                    X['NDREI2'] = (X['RE3'] - X['RE1']) / (X['RE3'] + X['RE1'])
                if "NDREI1" in self.indices_bands:
                    X['NDREI1'] = (X['RE2'] - X['RE1']) / (X['RE2'] + X['RE1'])
                if "SWIRI" in self.indices_bands:
                    X['SWIRI'] = X['SW1'] / X['BNR']
                if "CRSWIR" in self.indices_bands:
                    ### compute the CRSWIR index by following equation
                    X['CRSWIR'] = X['SW1'] / (X['NIR'] + ((X['SW2'] - X['NIR']) / (2185.7 - 864)) * (1610.4 - 864))
                if "NGRDI" in self.indices_bands:
                    X['NGRDI'] = (X['GRN'] - X['RED']) / (X['GRN'] + X['RED'])
                if "SRSWIR" in self.indices_bands:
                    X['SRSWIR'] = X['SW1'] / X['SW2']
                if "LWCI" in self.indices_bands:
                    X['LWCI'] = np.log(1 - (X['BNR'] - X['SW1'])) / -np.log(1 - (X['BNR'] - X['SW1']))
                ### drop all but the indices_bands columns and keep the order
                X = X[self.indices_bands + ['DOY']]
                ### drop dataframe rows which contain unreasonable values
                X.dropna(inplace=True)
                X = X[np.isfinite(X).all(axis=1)]
            
            ### if the smallest DOY value is greater than 365, subtract 365 from all DOY values
            ### this can happen because some values have been removed by the QAI criteria
            ### repeat this until the smallest DOY value is less than or equal to 365
            while X['DOY'].min() > 365:
                X['DOY'] -= 365

            ### the resulting dataframe could be empty because all values have been removed
            ### by qai criteria or by the index criteria
            if not X.empty:
                break # continue with preprocessing
            else:
                ### if X.empty evaluates to True
                ### sample a new idx value
                idx = np.random.randint(0, len(self.input_files))

        ### NORMALIZATION
        ### first apply the normalization, and only then apply the data augmentation
        ### code block inserted here as for test/val/predict flag, this is not relevant 
        ### because we do not apply data augmentation there
        ### check if fixed normalization parameters should be used
        if self.fixed_norm == 1:
            columns_to_normalize = [col for col in X.columns if col != 'DOY']
            X[columns_to_normalize] = X[columns_to_normalize].apply(lambda x: (x - self.overall_mean) / self.overall_std, axis=1)
        ### the following applies if fixed_norm == 0
        ### the idea behind this is that band and index values have the same range and are applicable across any region
        ### thus, we need them to be within the [0, 1] range (which is optimal for Deep Learning), 
        ### but a normalization using parameters from training is not really necessary
        else:
            ### here, we need to normalize the indices_bands columns only
            ### but it depends on the self.indices_bands list, 
            ### since the bands and many indices have their own normalization scheme
            ### all bands (BLU, GRN, RED, RE1, RE2, RE3, BNR, NIR, SW1, SW2) are in the [0, 10000] range
            ### divide them by 10000
            ### indices: 
            ### NDWI, CRSWIR, NDREI1 and NDREI2 are in the [-1, 1] range, 
            ### SWIRI is in the [0, 1] range, CLRE is in the [-1, 10] range, DSWI is in the [0, 6] range 
            ### we want all of them to be typically within [0, 1] range
            ### define normalization schemes for each column
            normalization_schemes = {
                'BLU': lambda x: x / 10000.0,
                'GRN': lambda x: x / 10000.0,
                'RED': lambda x: x / 10000.0,
                'RE1': lambda x: x / 10000.0,
                'RE2': lambda x: x / 10000.0,
                'RE3': lambda x: x / 10000.0,
                'BNR': lambda x: x / 10000.0,
                'NIR': lambda x: x / 10000.0,
                'SW1': lambda x: x / 10000.0,
                'SW2': lambda x: x / 10000.0,
                'NDWI': lambda x: (x + 1) / 2.0,  # Normalize from [-1, 1] to [0, 1]
                'CRSWIR': lambda x: x / 2.0,  # Normalize from [0, 2] to [0, 1]
                ### CRSWIR typically between 0 and 2
                ### healthy, moist vegetation: around .5 to 1
                ### dry or stressed vegetation: around 1 to 1.5
                ### bare soil: around 1 to 2
                ### <0 is physically meaningless for reflectance ratios
                ### >2 rarely happens in real-world conditions (possibly clouds, snow, shadows)
                'NDREI1': lambda x: (x + 1) / 2.0,  # Normalize from [-1, 1] to [0, 1]
                'NDREI2': lambda x: (x + 1) / 2.0,  # Normalize from [-1, 1] to [0, 1]
                'SWIRI': lambda x: x,  # Already in [0, 1] range
                'CLRE': lambda x: (x + 1) / 11.0,  # Normalize from [-1, 10] to [0, 1]
                'DSWI': lambda x: x / 6.0,  # Normalize from [0, 6] to [0, 1]
                'NGRDI': lambda x: (x + 1) / 2.0,  # Normalize from [-1, 1] to [0, 1]
                'SRSWIR': lambda x: x,  # Already in [0, 1] range (not strictly capped by 1, but roughly in healthy vegetation)
                'LWCI': lambda x: x  # Already in [0, 1] range (approximately)
            }
            # apply normalization to the indices_bands columns
            for col in self.indices_bands:
                if col in normalization_schemes:
                    X[col] = normalization_schemes[col](X[col])

        ### convert X (pandas dataframe) to numpy array
        ts = X.to_numpy()

        # get number of observations for further processing
        ts_length = ts.shape[0]

        ### this operation is necessary to convert the integers to floats
        ts = ts.astype(float)

        ### DATA AUGMENTATION
        ### add some time series augmentation (data augmentation) procedures
        ### but only for the train dataset
        if self.flag == "train":

            ### randomly jitter the DOY values by +/- 5 days (to simulate phenological differences)
            if np.random.rand() > 0.5:  # 50% chance to proceed with DOY jitter
                ### get random integers between +/-3 for all doy values
                doy_noise = np.random.randint(-3, 3, ts[:, -1].shape[0])
                ### get min and max before doy jitter (should not be changed, e.g. to avoid negative doy values)
                minimum = ts[:, -1].min()
                maximum = ts[:, -1].max()
                ### add the noise to the doy values
                ts[:, -1] += doy_noise
                ### clip to keep the correct range
                ts[:, -1] = np.clip(ts[:, -1], minimum, maximum)

            ### randomly add noise to the bands (to simulate sensor noise)
            ### add a bit of noise to every value with respect to standard deviation
            ### we want to add +/- .02 of the standard deviation for each of the columns
            if np.random.rand() > 0.5:  # 50% chance to proceed with observation jitter
                # compute the sd's
                std_devs = np.std(ts[:, :-1], axis=0)
                # generate noise that is 5% of each column's standard deviation
                # create noise for each element based on its column's standard deviation
                noise = np.random.normal(0, 0.02 * std_devs, ts[:, :-1].shape)
                # add this noise to ts[:, :-1]
                ts[:, :-1] = ts[:, :-1] + noise

            ### if smallest doy value is greater than 365, subtract 365 from all doy values
            ### because the time series always starts in year 1
            while np.min(ts[:, -1]) > 365:  ### check if the smallest doy value is greater than 365
                ts[:, -1] -= 365  ### subtract 365 from every doy value
            
            ### update ts_length (changes may have occurred)
            ts_length = ts.shape[0]

        ### day of year
        doy = np.zeros((self.win_size,), dtype=int)

        ### BOA reflectances
        ts_origin = np.zeros((self.win_size, ts.shape[1]-1)) # -1 because of doy column

        # in case of test data, we always use the last seq_len observations (not random sampling from sequence)
        if self.flag == "test" or self.flag == "val" or self.flag == "predict":
            if self.win_size > ts_length:
                ts_origin[:ts_length, :] = ts[:ts_length, :-1]
                doy[:ts_length] = np.squeeze(ts[:ts_length, -1])
            else:
                ts_origin[:self.win_size, :] = ts[:self.win_size, :-1]
                doy[:self.win_size] = np.squeeze(ts[-self.win_size:, -1])

            ### iterative prediction approach
            if self.flag == "predict":
                ### now set all values to 0 in both doy and ts_origin 
                ### from self.max_index onwards
                ### this way, we can test the iterative approach
                if self.max_index is not None:
                    doy[self.max_index:] = 0
                    ts_origin[self.max_index:, :] = 0.0

        ### for train and val data, we randomly sample a sequence of length seq_len from the input sequence
        else:
            ### case when the desired seq length is greater than the input sequence length
            if self.win_size > ts_length:
                ts_origin[:ts_length, :] = ts[:ts_length, :-1]
                doy[:ts_length] = np.squeeze(ts[:ts_length, -1])
            else:
                ts_origin[:self.win_size, :] = ts[:self.win_size, :-1]
                doy[:self.win_size] = np.squeeze(ts[:self.win_size, -1])

        ### REFORMATTING THE TIME SERIES
        ### at this point, we have a prepared time series (ts_origin) and DOY values
        ### we now want to change the time series to be suitable as input to the model without Fourier Transforms
        ### the model uses a fixed period, and (max) 4 years/seasons next to each other
        ### the goal is: 
        ### a time series ts_origin of shape [self.win_size, bands] and 
        ### a time series doy of shape [self.win_size]
        ### note that self.win_size is the maximum padded sequence length for every sample, which is provided by the user
        ### instead of end padding of the whole time series, we want end padding for each of the (up to) 4 years of the time series
        ### so that the goal is a ts_origin like: [(BOA reflectances first year), 0 , 0, 0, ..., (BOA reflectances second year), 0 , 0, 0, ...]
        ### where each year's values have the same length, and the DOY values are also padded accordingly
        ### the doy array should contain only values from 1 until 365 (each observation's day of the year)
        ### instead of the cumulative doy values (cumulative doy means 1 until 365 in first year, 366 until 730 in second year, etc.)
        ### the years can be separated by the doy values before adjusting them to "normal" doy values
        
        ### define the number of days in a year and the period length
        days_in_year = 365
        period = self.win_size // 4 # 4 years, so we divide the win_size by 4

        ### initialize lists to store the split data
        years = []
        doy_years = []

        ### split the original time series and DOY values into years
        for i in range(4):
            start_day = i * days_in_year + 1
            end_day = (i + 1) * days_in_year
            mask = (doy >= start_day) & (doy <= end_day)
            year_data = ts_origin[mask]
            doy_data = doy[mask] - start_day + 1  # Adjust DOY to be within 1 to 365

            ### if there are no observations for the current year, create an empty array
            if len(year_data) == 0:
                year_data = np.zeros((period, ts_origin.shape[1]), dtype=ts_origin.dtype)
                doy_data = np.zeros(period, dtype=doy.dtype)
            years.append(year_data)
            doy_years.append(doy_data)

        ### ensure that each year has the same length by padding with zeros if necessary
        for i in range(4):
            if len(years[i]) < period:
                padding_length = period - len(years[i])
                years[i] = np.pad(years[i], ((0, padding_length), (0, 0)), mode='constant', constant_values=0)
                doy_years[i] = np.pad(doy_years[i], (0, padding_length), mode='constant', constant_values=0)
            else:
                years[i] = years[i][:period]
                doy_years[i] = doy_years[i][:period]

        ### shuffle the years and DOYs if self.flag is "train"
        if self.flag == "train":
            ### identify non-empty years
            non_empty_indices = [i for i in range(4) if np.any(years[i])]
            ### identify empty years (no values except padded zeros)
            empty_indices = [i for i in range(4) if not np.any(years[i])]
            
            ### shuffle non-empty years
            ### reasoning: the last year might not be a full year of observations
            ### and we don't want to use half-empty years in the middle of the sequence
            ### (although, admittedly, it will likely not have a large impact)
            if len(non_empty_indices) > 1:
                np.random.shuffle(non_empty_indices)
                shuffled_years = [years[i] for i in non_empty_indices] + [years[i] for i in empty_indices]
                shuffled_doy_years = [doy_years[i] for i in non_empty_indices] + [doy_years[i] for i in empty_indices]
                years[:] = shuffled_years
                doy_years[:] = shuffled_doy_years

        ### convert lists to numpy arrays
        years = np.array(years)
        doy_years = np.array(doy_years)

        ### now we have two arrays of shape [num_years, period, bands] and [num_years, period]
        ### we want to concatenate them to get arrays of shape [self.win_size, bands] and [self.win_size]
        years_reshaped = years.reshape(-1, years.shape[2])
        doy_years_reshaped = doy_years.reshape(-1)
        
        ### replace the original ts_origin and doy with the new ones
        ts_origin = years_reshaped
        doy = doy_years_reshaped

        ### return a tuple of numpy arrays
        return (ts_origin, doy)