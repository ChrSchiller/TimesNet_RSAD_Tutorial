import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import random

### function that computes the mean values of each band for each sample
### note that the function computes the mean of the sample bands' means, 
### as a more parallelizable approach
### we consider it a good approximation anyway
def calculate_normalization_params(train_files, num_workers, use_qai=0, indices_bands=[]):
    means = []
    stds = []

    def process_file(file, use_qai=0, indices_bands=[]):
        df = pd.read_csv(file)

        ### drop the rows with doy == 0 (if any)
        df = df.loc[df['DOY'] != 0]
        ### do not consider the doy nor the date column
        df = df.drop(columns=['DOY', 'date'])

        if use_qai == 1:
            ### convert QAI column (integer) to 15-bit binary
            df['QAI_binary'] = df['QAI'].apply(lambda x: format(x, '015b'))
            ### convert the QAI_binary column to string
            df['QAI_binary'] = df['QAI_binary'].astype(str)
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
            df['valid'] = df['QAI_binary'].apply(lambda x: x[14])
            df['cloud'] = df['QAI_binary'].apply(lambda x: x[12:14])
            df['cloud_shadow'] = df['QAI_binary'].apply(lambda x: x[11])
            df['snow'] = df['QAI_binary'].apply(lambda x: x[10])
            df['water'] = df['QAI_binary'].apply(lambda x: x[9])
            df['aerosol'] = df['QAI_binary'].apply(lambda x: x[7:9])
            df['subzero'] = df['QAI_binary'].apply(lambda x: x[6])
            df['saturation'] = df['QAI_binary'].apply(lambda x: x[5])
            df['high_sun_zenith'] = df['QAI_binary'].apply(lambda x: x[4])
            df['illumination'] = df['QAI_binary'].apply(lambda x: x[2:4])
            df['slope'] = df['QAI_binary'].apply(lambda x: x[1])
            df['water_vapor'] = df['QAI_binary'].apply(lambda x: x[0])
            ### remove rows with the following features: 
            ### valid == 1, cloud == 10 or 11, cloud_shadow == 1, snow == 1, water == 1, aerosol == 10 or 11, 
            ### subzero == 1, saturation == 1, high_sun_zenith == 1, illumination == 10 or 11, water_vapor == 1
            df = df[(df['valid'] == "0") & (df['cloud'] != "10") & (df['cloud'] != "11") & 
                    (df['cloud_shadow'] == "0") & (df['snow'] == "0") & (df['water'] == "0") & 
                    (df['aerosol'] != "10") & (df['aerosol'] != "11") & (df['subzero'] == "0") & 
                    (df['saturation'] == "0") & (df['high_sun_zenith'] == "0") & 
                    (df['illumination'] != "10") & (df['illumination'] != "11")]
            ### drop all QAI-related columns
            df = df.drop(columns=['QAI', 'QAI_binary', 'valid', 'cloud', 'cloud_shadow', 'snow', 'water', 'aerosol', 'subzero', 'saturation', 'high_sun_zenith', 'illumination', 'slope', 'water_vapor'])
        ### else: if use_qai == 0
        else:
            ### if the QAI column exists, drop it
            if 'QAI' in df.columns:
                df = df.drop(columns=['QAI'])
        
        ### list of indices to check
        indices_to_check = ["DSWI", "NDWI", "CLRE", "NDREI2", "NDREI1", "SWIRI", "CRSWIR", "NGRDI", "SRSWIR", "LWCI"]
        ### check if at least one of the indices is in indices_bands
        if any(index in indices_bands for index in indices_to_check):
            ### compute the indices
            if "DSWI" in indices_bands:
                df['DSWI'] = (df['BNR'] + df['GRN']) / (df['SW1'] + df['RED'])
                df = df[(df['DSWI'] >= -1) & (df['DSWI'] <= 5)]
            if "NDWI" in indices_bands:
                df['NDWI'] = (df['GRN'] - df['BNR']) / (df['GRN'] + df['BNR'])
                df = df[(df['NDWI'] >= -1) & (df['NDWI'] <= 1)]
            if "CLRE" in indices_bands:
                df['CLRE'] = (df['RE3'] / df['RE1']) - 1
                df = df[(df['CLRE'] >= -1) & (df['CLRE'] <= 10)]
            if "NDREI2" in indices_bands:
                df['NDREI2'] = (df['RE3'] - df['RE1']) / (df['RE3'] + df['RE1'])
                df = df[(df['NDREI2'] >= -1) & (df['NDREI2'] <= 1)]
            if "NDREI1" in indices_bands:
                df['NDREI1'] = (df['RE2'] - df['RE1']) / (df['RE2'] + df['RE1'])
                df = df[(df['NDREI1'] >= -1) & (df['NDREI1'] <= 1)]
            if "SWIRI" in indices_bands:
                df['SWIRI'] = df['SW1'] / df['BNR']
                df = df[(df['SWIRI'] >= 0) & (df['SWIRI'] <= 1)] # exception: SWIRI must be positive
            if "CRSWIR" in indices_bands:
                df['CRSWIR'] = df['SW1'] / (df['NIR'] + ((df['SW2'] - df['NIR']) / (2185.7 - 864)) * (1610.4 - 864))
                df = df[(df['CRSWIR'] >= -1) & (df['CRSWIR'] <= 1)]
            if "NGRDI" in indices_bands:
                df['NGRDI'] = (df['GRN'] - df['RED']) / (df['GRN'] + df['RED'])
                df = df[(df['NGRDI'] >= -1) & (df['NGRDI'] <= 1)]
            if "SRSWIR" in indices_bands:
                df['SRSWIR'] = df['SW1'] / df['SW2']
                df = df[(df['SRSWIR'] >= 0) & (df['SRSWIR'] <= 2)] ### upper cap 1 in healthy forest, we allow 2
            if "LWCI" in indices_bands:
                df['LWCI'] = np.log(1 - (df['BNR'] - df['SW1'])) / -np.log(1 - (df['BNR'] - df['SW1']))
            ### drop all but the indices_bands columns and keep the order
            df = df[indices_bands]
            ### drop df rows which contain unreasonable values
            df.dropna(inplace=True)
            df = df[np.isfinite(df).all(axis=1)]

        return df.mean(skipna=True).values, df.std(skipna=True).values

    ### get mean of each sample for each band separately
    ### see note above: in the end, we take the mean of the band means, 
    ### which is not the same as the mean of all observations
    ### but much faster to process
    results = Parallel(n_jobs=num_workers)(delayed(process_file)(file, use_qai, indices_bands) for file in train_files)

    for mean, std in results:
        means.append(mean)
        stds.append(std)

    ### get length of means to check if it worked
    print("Number of samples: ", len(means))

    ### get the mean of the band means
    overall_mean = np.nanmean(means, axis=0)
    overall_std = np.nanmean(stds, axis=0)

    return overall_mean, overall_std


def store_train_val_test(root_dir, num_workers=1, fixed_norm=0, info="", seed=123, use_qai=0, indices_bands=[]):
    ### set seed
    random.seed(seed)
    np.random.seed(seed)

    ### read metadata
    meta = pd.read_csv(root_dir + '/meta/metadata.csv', sep=",")

    ### create directory if necessary
    if not os.path.exists(os.path.join(root_dir + '/split_data')):
        os.mkdir(os.path.join(root_dir + '/split_data'))
    
    ### change type to avoid mistakes
    meta.dataset = meta.dataset.astype(str)

    ### define testdat as all meta rows from the May dataset
    testdat = meta.loc[(meta['dataset'] == 'may')] 

    ### select only the rows from the lux dataset for training/validation
    meta = meta.loc[(meta['dataset'] == 'lux')] 

    ### remove all frac_coniferous != 1 observations from meta
    ### this could be changed by commenting out -> all forest types (including deciduous) would be used
    meta = meta.loc[meta['frac_coniferous'] == 1]

    ### split the data into training and validation datasets
    traindat, valdat = train_test_split(meta, test_size=.2, random_state=seed)

    ### print number of samples in each set
    print("\nTraining Set Size: ", len(traindat))
    print("\nValidation Set Size: ", len(valdat))
    print("\nTest Set Size: ", len(testdat))

    ### save the split data incl args.info as for the mean and sd parameters
    traindat.to_csv(root_dir + '/split_data/train_' + str(info) + '.csv', index=False)
    valdat.to_csv(root_dir + '/split_data/val_' + str(info) + '.csv', index=False)
    testdat.to_csv(root_dir + '/split_data/test_' + str(info) + '.csv', index=False)

    if fixed_norm == 1:
        ### continue with this only if mean_df and std_df are not already present
        if os.path.exists(root_dir + '/split_data/train_overall_mean_indices_bands_' + str(info) + '.csv') and \
           os.path.exists(root_dir + '/split_data/train_overall_std_indices_bands_' + str(info) + '.csv'):
            print("Normalization parameters already computed. Skipping...")
        else:
            print("Computing fixed normalization parameters...")
            ### compute fixed normalization parameters
            train_files = [root_dir + "/" + f + ".csv" for f in traindat['plotID']]
            overall_mean, overall_std = calculate_normalization_params(train_files, num_workers, use_qai, indices_bands)
            ### convert to DataFrame
            mean_df = pd.DataFrame(overall_mean, columns=['mean'])
            std_df = pd.DataFrame(overall_std, columns=['std'])

            ### save as csv files
            mean_df.to_csv(root_dir + '/split_data/train_overall_mean_indices_bands_' + str(info) + '.csv', index=False)
            std_df.to_csv(root_dir + '/split_data/train_overall_std_indices_bands_' + str(info) + '.csv', index=False)