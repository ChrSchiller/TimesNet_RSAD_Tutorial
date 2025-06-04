### import necessary libraries
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
print(torch.version.cuda)
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from utils.store_train_val_test import store_train_val_test
import random
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='anomaly_detection',
                        help='task name, options:[anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='training required? 0 = No, 1 = Yes')
    parser.add_argument('--is_testing', type=int, required=False, default=1, help='testing desired? 0 = No, 1 = Yes')
    parser.add_argument('--is_predict_iter', type=int, required=False, default=0, help='iterative inference desired? 0 = No, 1 = Yes')
    parser.add_argument('--model_id', type=str, required=False, default='RsTs', help='model id: only RsTs is supported for now')
    parser.add_argument('--model', type=str, required=False, default='TimesNet_RSAD',
                        help='model name, options: [TimesNet_RSAD]')
    parser.add_argument('--info', type=str, default='', help='additional info')
    ### add seed
    parser.add_argument('--seed', type=int, default=123, help='seed for random number generator')

    # data loader
    ### path to the saved model
    parser.add_argument('--model_path', type=str, required=False, default='None', help='path to directory of saved model (optional)')
    parser.add_argument('--root_path', type=str, default='', help='root path of the data file')
    ### add fixed normalization as 0 vs 1 argument (yes or no)
    parser.add_argument('--fixed_norm', type=int, default=1, help='use fixed normalization: 0 = no, 1 = yes')
    ### optionally, pass normalization values for the bands as a list of floats (mean and std separately)
    ### if fixed_norm == 1 and these two arguments are provided, the parameters are not computed, 
    ### but taken from the command line
    ### if fixed_norm == 1 and these two arguments are not provided, the parameters are computed
    parser.add_argument('--norm_mean', nargs='+', type=float, required=False, default=[],
                    help='Optional list of normalization values (mean and std) for indices_bands')
    parser.add_argument('--norm_std', nargs='+', type=float, required=False, default=[],
                    help='Optional list of normalization values (mean and std) for indices_bands')
    ### parse a list of indices or bands to be used
    parser.add_argument('--indices_bands', nargs='+', default=['CRSWIR'],
                        help='List of indices to be used as input data, e.g. [BLU, GRN, RED, RE1, RE2, RE3, BNR, NIR, SW1, SW2]')
    ### add 0 or 1 if QAI flags from FORCE should be used
    parser.add_argument('--use_qai', type=int, default=1, help='use QAI flags: 0 = no, 1 = yes')
    ### the data argument can only be used with one value in the current version, 
    ### but we might add some other datasets in the future, hence we keep it
    parser.add_argument('--data', type=str, required=False, default='RsTs', help='dataset type')
    ### we keep the seq_len argument for the same reason as data argument
    ### it is used to define the input sequence length for the model
    ### in the current version, it is always 200, but we might change it in the future
    parser.add_argument('--seq_len', type=int, default=200, help='input sequence length')

    # model define
    parser.add_argument('--num_kernels', type=int, default=8, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=10, help='encoder input size', required=False)
    parser.add_argument('--c_out', type=int, default=10, help='output size', required=False)
    ### note that d_model also defines the number of output channels of the token embedding
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=8, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--embed', type=str, default='learnedsincos',
                        help='time features encoding, options:[timeF, fixed, learned, learnedsincos]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained model')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=20, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # forecasting task
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')

    ### parse the arguments
    args = parser.parse_args()
    args.indices_bands = args.indices_bands if args.indices_bands else []

    ### change enc_in and c_out to the number of indices and bands
    args.enc_in = len(args.indices_bands)
    args.c_out = len(args.indices_bands)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    ### add "_" + fix_seed to args.info
    args.info = args.info + "_seed" + str(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    ### check (assert) if norm_mean and norm_std are both provided
    ### they can either be both provided (both not None) or both not provided (both None)
    if (args.norm_mean is None and args.norm_std is not None) or (args.norm_mean is not None and args.norm_std is None):
        raise ValueError("Both norm_mean and norm_std must be provided or both must be None.")

    if args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification

    ### prepare the datasets
    if args.model_id == "RsTs" and args.is_training == 1:
        ### root_path/meta/metadata.csv is always the path to metadata file, 
        ### so we only need to provide the root_path
        store_train_val_test(args.root_path, args.num_workers, args.fixed_norm, args.info, args.seed, args.use_qai, args.indices_bands)

    if args.is_training == 1:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_sl{}_dm{}_el{}_df{}_eb{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.d_model,
                args.e_layers,
                args.d_ff,
                args.embed,
                args.info, 
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting, args.model_id, args.pretrained, args.root_path)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, args.model_id)
            torch.cuda.empty_cache()
    elif args.is_testing == 1:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_dm{}_el{}_df{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.d_model,
            args.e_layers,
            args.d_ff,
            args.embed,
            args.info, 
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, args.model_id, test=1)
        torch.cuda.empty_cache()

    if args.is_predict_iter == 1:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_dm{}_el{}_df{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.d_model,
            args.e_layers,
            args.d_ff,
            args.embed,
            args.info, 
            ii)

        exp = Exp(args)
        print('>>>>>>>predicting iteratively : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict_iteratively(setting, args.model_id, args.root_path)
        torch.cuda.empty_cache()
        print("Iterative prediction finished")