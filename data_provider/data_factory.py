from data_provider.data_loader import RsTsLoader
from torch.utils.data import DataLoader

data_dict = {
    'RsTs': RsTsLoader # our dataset -> our DataLoader
}


def data_provider(args, flag, max_index=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'predict':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    if args.task_name == 'anomaly_detection':
        drop_last = False

        ### Create the dataset and dataloader for anomaly detection task
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
            max_index=max_index, 
            fixed_norm=args.fixed_norm, 
            use_qai=args.use_qai,
            info=args.info, 
            indices_bands=args.indices_bands, 
            norm_mean=args.norm_mean,
            norm_std=args.norm_std,
        )
        
        ### data_set is the Dataset class, which is then used in the DataLoader
        print(flag, len(data_set))

        ### Create the DataLoader for the dataset
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    
    ### I commented out the rest of the script because we don't need it
    ### and it is confusing to have so much code that is not used
    # elif args.task_name == 'classification':
    #     drop_last = False
    #     data_set = Data(
    #         root_path=args.root_path,
    #         flag=flag,
    #     )

    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last,
    #         collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    #     )
    #     return data_set, data_loader
