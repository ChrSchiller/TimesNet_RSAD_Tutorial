from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

### function to adjust doy values for a single sample
def adjust_doy_values(doy_values):
    ### get the positions of the zeros
    zero_positions = np.where(doy_values == 0)[0]
    
    ### remove all zeros
    non_zero_doy_values = doy_values[doy_values != 0]
    
    ### adjust the doy values
    for i in range(1, len(non_zero_doy_values)):
        while non_zero_doy_values[i] <= non_zero_doy_values[i - 1]:
            non_zero_doy_values[i] += 365
    
    ## re-introduce the zeros at the stored index positions
    adjusted_doy_values = np.ones_like(doy_values)
    adjusted_doy_values[zero_positions] = 0
    non_zero_index = 0
    for i in range(len(adjusted_doy_values)):
        if adjusted_doy_values[i] != 0:
            adjusted_doy_values[i] = non_zero_doy_values[non_zero_index]
            non_zero_index += 1
    
    return adjusted_doy_values

### class definition for anomaly detection experiment
class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, max_index=None):
        data_set, data_loader = data_provider(self.args, flag, max_index)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, model_id=None):
        total_loss = []
        single_losses = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                if model_id == "RsTs":
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)

                ### loss is only computed on batch level, not on sample level
                total_loss.append(loss)
        if model_id == "RsTs":
            # get the loss values for each sample
            single_losses = total_loss
            # single_losses is a list of 0-d torch tensors
            single_losses = [i.item() for i in single_losses]
            total_loss = np.average(total_loss)
            self.model.train()
            return total_loss, single_losses
        else:
            total_loss = np.average(total_loss)
            self.model.train()

            return total_loss

    ### train method definition
    def train(self, setting, model_id=None, pretrained=False, base_path="."):
        ### this is the sample time series preparation code
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        ### define model path
        print('loading model')
        ### if model_path is given, use it
        if self.args.model_path != "None":
            print("Loading model from: ", self.args.model_path)
            model_path = self.args.model_path
        else:
            ### otherwise, load the model using the code below
            model_path = os.path.join(base_path, 'models', setting) # setting is the "model name"

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(os.path.join(model_path, "results")):
            os.makedirs(os.path.join(model_path, "results"), exist_ok=True)

        if pretrained:
            print('loading pretrained model')
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pth')))
            ### save a copy of that model to store it in the checkpoints folder
            torch.save(self.model.state_dict(), os.path.join(os.path.join(model_path, 'checkpoint_only_pretrain.pth')))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss_list = []
        val_loss_list = []

        ### print one example of a single sample of a train_loader output
        ### just for checking if everything works well
        ### and the values are in a reasonable range

        ### get an iterator for the train_loader
        train_loader_iter = iter(train_loader)

        ### draw one sample (one batch) from the train_loader
        batch_x, batch_x_mark = next(train_loader_iter)

        ### print the first sample of the batch
        print("Example of a single sample of a train_loader output:")
        print("batch_x (S2 observations/indices):")
        print(batch_x[0])
        print("batch_x_mark (DOY):")
        print(batch_x_mark[0])

        ### print one example of a single sample of a vali_loader output
        ### just for checking if everything works well
        ### and the values are in a reasonable range

        ### get an iterator for the train_loader
        vali_loader_iter = iter(vali_loader)

        ### draw one sample (one batch) from the train_loader
        batch_x, batch_x_mark = next(vali_loader_iter)

        ### print the first sample of the batch
        print("Example of a single sample of a vali_loader output:")
        print("batch_x (S2 observations/indices):")
        print(batch_x[0])
        print("batch_x_mark (DOY):")
        print(batch_x_mark[0])

        ### print one example of a single sample of a test_loader output
        ### just for checking if everything works well
        ### and the values are in a reasonable range

        ### get an iterator for the test_loader
        test_loader_iter = iter(test_loader)

        ### draw one sample (one batch) from the test_loader
        batch_x, batch_x_mark = next(test_loader_iter)

        ### print the first sample of the batch
        print("Example of a single sample of a test_loader output:")
        print("batch_x (S2 observations/indices):")
        print(batch_x[0])
        print("batch_x_mark (DOY):")
        print(batch_x_mark[0])

        ### print the number of trainable parameters of the model
        print("Number of trainable parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                ### get outputs from the model (this is the forward pass)
                outputs = self.model(batch_x, batch_x_mark, None, None)

                ### calculate loss
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                ### backward pass and optimization
                ### this is the backpropagation step
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            ### add to train_loss_list to be able to plot the training loss
            train_loss_list.append(train_loss)

            ### compute loss on validation dataset after each epoch
            vali_loss, single_losses = self.vali(vali_data, vali_loader, criterion, model_id)

            ### add to val_loss_list to be able to plot the validation loss
            val_loss_list.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            ### adjust the learning rate by lr scheme
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            ### create and save the training and validation loss plot (after each epoch)
            epochs_range = range(epoch + 1)
            plt.figure(figsize=(15, 15))
            plt.plot(epochs_range, train_loss_list, label='Training Loss')
            plt.plot(epochs_range, val_loss_list, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            ### increase all axes labels and titles as well as legend font size
            plt.xlabel('Epoch', fontsize=20)
            plt.ylabel('Loss', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
            plt.savefig(os.path.join(model_path, 'train_val_loss.pdf'),
                        bbox_inches='tight')

            ### save the plot information to disk (train_loss_list and val_loss_list)
            np.savetxt(os.path.join(model_path, 'train_loss_list.txt'), train_loss_list, fmt='%f', delimiter='\n')
            np.savetxt(os.path.join(model_path, 'val_loss_list.txt'), val_loss_list, fmt='%f', delimiter='\n')

        best_model_path = model_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        ### save the single losses (batch-level) to disk
        ### convert to a format suitable for saving as txt file
        with open(os.path.join(model_path, 'batch_val_losses.txt'), 'w') as f:
            for loss in single_losses:
                f.write(f"{loss}\n")

        return self.model

	### test method definition
    def test(self, setting, model_id=None, test=0, indices_bands=[]):

        ### load model
        print('loading model')
        ### if model_path is given, load the model from there
        if self.args.model_path != "None":
            print("Loading model from: ", self.args.model_path)
            model_path = self.args.model_path
        else:
            ### otherwise, load the model using the code below
            model_path = os.path.join(self.args.root_path, 'models', setting)
            
        ### create model path if it does not exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(os.path.join(model_path, "results")):
            os.makedirs(os.path.join(model_path, "results"), exist_ok=True)

        ### load specified model
        best_model_path = model_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        ### get validation and test data and DataLoaders ready
        test_data, test_loader = self._get_data(flag='test')
        vali_data, vali_loader = self._get_data(flag='val')

        attens_energy = []
        attens_energy_each_band = []
        folder_path = model_path + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        ### also mkdir with results folder for test results
        if not os.path.exists(folder_path + "preds"):
            os.makedirs(folder_path + "preds")

        ### set model to eval mode
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        ### statistic on the vali set
        with torch.no_grad():
            ### for loop over batches of data in validation dataset
            for i, (batch_x, batch_x_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # reconstruction
                outputs = self.model(batch_x, batch_x_mark, None, None)
                
                # criterion
                ### note that the DOY of each observation is always known to the model, even in test phase
                ### so it does not have to predict it
                
                ### note that since we do not use masking but end padding without a mask in training, 
                ### the model will predict the entire sequence, even the padded part
                ### this is not a problem, as we can just ignore the padded part
                ### but we have to exclude the padded part from loss calculation
                ### this is done by multiplying the loss with a mask that is determined by 0 values in batch_x
                mask = (batch_x != 0).float()  # assuming 0 is the padding value
                masked_outputs = outputs * mask
                masked_batch_x = batch_x * mask

                ### get anomaly scores by criterion
                score_each_band = self.anomaly_criterion(masked_batch_x, masked_outputs)
                score = torch.mean(self.anomaly_criterion(masked_batch_x, masked_outputs), dim=-1)

                score = score.detach().cpu().numpy()
                score_each_band = score_each_band.detach().cpu().numpy()
                attens_energy.append(score)
                attens_energy_each_band.append(score_each_band)
            
            ### check if attens_energy is a list of scalars
            ### (this is the case if using only one band/index)
            if all(a.ndim == 0 for a in attens_energy):
                ### convert each scalar to a one-dimensional array
                attens_energy = [np.expand_dims(a, axis=0) for a in attens_energy]
                attens_energy = np.concatenate(attens_energy, axis=0)
                attens_energy_each_band = [np.expand_dims(a, axis=0) for a in attens_energy_each_band]
                attens_energy_each_band = np.concatenate(attens_energy_each_band, axis=0)
            else: # not a list of scalars
                attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
                attens_energy_each_band = np.concatenate(attens_energy_each_band, axis=0).reshape(-1)
            vali_energy = np.array(attens_energy)
            vali_energy_each_band = np.array(attens_energy_each_band)

            ### ensure that vali_energy_each_band has the correct shape
            ### to save a multi-column txt file to disk
            if len(vali_energy_each_band.shape) == 1:
                ### if it's a 1D array, reshape it to (num_samples, 1)
                vali_energy_each_band = vali_energy_each_band.reshape(-1, 1)

            ### write to disk all validation losses (for follow-up statistics)
            np.savetxt(os.path.join(model_path, "single_val_losses.txt"), vali_energy, fmt='%f', delimiter=',')
            np.savetxt(os.path.join(model_path, "single_val_losses_each_band.txt"), vali_energy_each_band, fmt='%f', delimiter=',')

        # (2) prepare the predictions (test dataset)
        attens_energy = []
        attens_energy_each_band = []                    

        ### implementing the same loop as in original code
        ### but with batch_x_mark (doy) included
        ### for loop over batches of data in test dataset
        for i, (batch_x, batch_x_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            
            ### reconstruction
            outputs = self.model(batch_x, batch_x_mark, None, None)

            ### note that since we do not use masking but end padding without a mask in training, 
            ### the model will predict the entire sequence, even the padded part
            ### this is not a problem, as we can just ignore the padded part
            ### but we have to exclude the padded part from loss calculation
            ### this is done by multiplying the loss with a mask that is determined by 0 values in batch_x
            mask = (batch_x != 0).float()  # assuming 0 is the padding value
            outputs = outputs * mask
            batch_x = batch_x * mask
            
            # criterion
            ### acquiring an anomaly score for each band helps to identify the most important bands
            score_each_band = self.anomaly_criterion(batch_x, outputs)
            score_each_band = score_each_band.detach().cpu().numpy()
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            attens_energy_each_band.append(score_each_band)

            ### define base_columns for saving the results
            base_columns = indices_bands + ['doy']

            ### generate the column names list with '_pred' suffix for predictions and add 'score' at the end
            column_names = base_columns + [f"{col}_pred" for col in base_columns if col != 'doy'] + ['anomaly_score']
            ### convert the list of column names to a single string separated by spaces
            header_string = ' '.join(column_names)

            ### loop through the batch, 
            ### concatenate original values (batch_x, batch_x_mark) with the predictions (outputs)
            ### and write them to a file in os.path.join(folder_path, "preds")
            ### the file name should be the file name of the original file (test_data)
            for j in range(batch_x.shape[0]):
                ### write the predictions to a file
                np.savetxt(os.path.join(folder_path, "preds", test_data.input_files.plotID.iloc[i * self.args.batch_size + j] + ".txt"), 
                        np.concatenate(
                            [batch_x[j].detach().cpu().numpy(), 
                                batch_x_mark[j].detach().cpu().numpy().reshape(-1, 1),  # Reshape to (200, 1)
                                outputs[j].detach().cpu().numpy(),
                                score[j].reshape(-1, 1)],  # Reshape to (200, 1)
                                axis=1), 
                                header=header_string, 
                                )
                np.savetxt(os.path.join(folder_path, "preds", test_data.input_files.plotID.iloc[i * self.args.batch_size + j] + "_score_each_band.txt"), 
                            score_each_band[j], 
                            header=' '.join(base_columns[:-1]) # Exclude 'doy' from the header
                            )

    
    def predict_iteratively(self, setting, model_id=None, base_path=".", indices_bands=[]):

        ### load model
        print('loading model')
        ### if model_path is given, load the model from there
        if self.args.model_path != "None":
            print("Loading model from: ", self.args.model_path)
            model_path = self.args.model_path
        else:
            ### otherwise, load the model using the code below
            model_path = os.path.join(self.args.root_path, 'models', setting)
            
        ### load specified model
        best_model_path = model_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(os.path.join(model_path, "preds_iter")):
            os.makedirs(os.path.join(model_path, "preds_iter"), exist_ok=True)

        ### save all predictions to disk, 
        ### or just the latest one?
        ### switch to False after first iteration
        ### that's because we need all predictions once, but then only have to add the latest (new) one
        save_all = True

        for max_index in range(60, int(self.args.seq_len)):
            print("Predictions for time series until index: ", max_index, "of ", str(int(self.args.seq_len)))
            
            folder_path = model_path + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            ### also mkdir with results folder for test results
            if not os.path.exists(folder_path + "preds_iter/" + str(max_index)):
                os.makedirs(folder_path + "preds_iter/" + str(max_index))

            ### set model to eval mode
            self.model.eval()
            ### define the criterion for anomaly detection
            self.anomaly_criterion = nn.MSELoss(reduce=False)

            ### initialize the predict_loader
            predict_data, predict_loader = self._get_data(flag='predict', max_index=max_index)
            
            ### get predictions and extract latest anomaly score (the only one we are interested in)
            for i, (batch_x, batch_x_mark) in enumerate(predict_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, batch_x_mark, None, None)

                ### note that since we do not use masking but end padding without a mask in training, 
                ### the model will predict the entire sequence, even the padded part
                ### this is not a problem, as we can just ignore the padded part
                ### but we have to exclude the padded part from loss calculation
                ### this is done by multiplying the loss with a mask that is determined by 0 values in batch_x
                mask = (batch_x != 0).float()  # assuming 0 is the padding value
                outputs = outputs * mask
                batch_x = batch_x * mask
                
                ### criterion
                ### acqiuring an anomaly score for each band helps to identify the most important bands
                score_each_band = self.anomaly_criterion(batch_x, outputs)
                score_each_band = score_each_band.detach().cpu().numpy()
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()

                ### define base_columns for saving the results
                base_columns = indices_bands + ['doy']

                ### we need to adjust the doy values so that we can recover the correct date later
                ### the target array is batch_x_mark, as it contains the doy values
                batch_x_mark = np.apply_along_axis(adjust_doy_values, 1, batch_x_mark.detach().cpu().numpy())

                ### generate the column names list with '_pred' suffix for predictions and add 'score' at the end
                column_names = base_columns + [f"{col}_pred" for col in base_columns if col != 'doy'] + ['anomaly_score']
                ### convert the list of column names to a single string separated by spaces
                header_string = ' '.join(column_names)
                ### add bands + "band_anomaly_score" to the header string: BLU_band_anomaly_score, GRN_band_anomaly_score, ...
                header_string += ' ' + ' '.join([f"{col}_band_anomaly_score" for col in base_columns[:-1]])
                ### add plotID header
                header_string += ' ' + 'plotID'

                if save_all: 
                    ### loop through the batch, 
                    ### concatenate original values (batch_x, batch_x_mark) with the predictions (outputs)
                    ### and write them to a file in os.path.join(folder_path, "preds")
                    ### the file name should be the file name of the original file (test_data)
                    for j in range(batch_x.shape[0]):
                        ### write the predictions to a file
                        np.savetxt(os.path.join(folder_path, "preds_iter", str(max_index), predict_data.input_files.plotID.iloc[i * self.args.batch_size + j] + ".txt"), 
                                np.concatenate(
                                    [batch_x[j, 0:max_index, :].detach().cpu().numpy(), 
                                    batch_x_mark[j, 0:max_index].reshape(-1, 1),  # Reshape to (200, 1) # .detach().cpu().numpy()
                                    outputs[j, 0:max_index, :].detach().cpu().numpy(),
                                    score[j, 0:max_index].reshape(-1, 1),  # Reshape to (200, 1)
                                    score_each_band[j, 0:max_index],
                                    np.full((max_index, 1), predict_data.input_files.plotID.iloc[i * self.args.batch_size + j])
                                    ], 
                                    axis=1), 
                                    header=header_string, 
                                    fmt='%s'
                                    )
                else:
                    ### we only need the latest anomaly score (overall + each band)
                    score = score[:, max_index-1]
                    score_each_band = score_each_band[:, max_index-1, :]

                    ### loop through the batch, 
                    ### concatenate original values (batch_x, batch_x_mark) with the predictions (outputs)
                    ### and write them to a file in os.path.join(folder_path, "preds")
                    ### the file name should be the file name of the original file (test_data)
                    for j in range(batch_x.shape[0]):
                        ### write the predictions to a file
                        np.savetxt(os.path.join(folder_path, "preds_iter", str(max_index), predict_data.input_files.plotID.iloc[i * self.args.batch_size + j] + ".txt"), 
                                np.concatenate(
                                    [batch_x[j, max_index-1, :].detach().cpu().numpy().reshape(1, -1), 
                                    batch_x_mark[j, max_index-1].reshape(-1, 1), # .detach().cpu().numpy()
                                    outputs[j, max_index-1, :].detach().cpu().numpy().reshape(1, -1),
                                    score[j].reshape(-1, 1), 
                                    score_each_band[j].reshape(1, -1), 
                                    np.array([[predict_data.input_files.plotID.iloc[i * self.args.batch_size + j]]])
                                    ],  
                                    axis=1), 
                                    header=header_string, 
                                    fmt='%s'
                                    )
                    
            ### only save all results in first iteration: 
            save_all = False
