##### This script takes as input the predictions of the model and the original observations
##### it then maps both in one line plot for each pixel and each band/index
##### and creates another plot showing the reconstruction errors (anomaly scores) for each band/index
##### these two plots are put into one figure with two panels, which is saved as a PDF file

### specify your packages
my_packages <- c('ggplot2', 'tidyr', 'scales', 'ggthemes', 'future', 'stringr', 'patchwork',
                 'future.apply', 'furrr', 'purrr', 'magick', 'dplyr', 'lubridate')
### extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , 'Package'])]
### install not installed packages
if(length(not_installed)) install.packages(not_installed)

### random seed
set.seed(123)

### imports
require(ggplot2)
require(tidyr)
require(scales)
require(ggthemes)
library(future)
library(future.apply)
library(furrr)
library(purrr)
library(magick)
library(dplyr)
library(lubridate)
require(stringr)
library(patchwork)


### define function that can be used in future_map, iterating through preds
### and using BASE_PATH, MODEL_NAME, MSE, meta
### this is the function for the undisturbed and infested dataset (depending on the "infested" parameter)
plot_example_time_series_and_anomaly_score <- function(iter, preds, base_path, model_dir_name, mse, meta, bands, infested = FALSE){

  ### read dataset containing the predictions
  dat <- read.table(paste0(base_path, '/models/', model_dir_name, '/preds/', preds[iter]), 
                      header = FALSE, skip = 1) # skip the first line which is a header
  ###define suitable column names
  colnames(dat) <- c(bands, 'doy', paste0(bands, "_pred"), 'anomaly_score')
  ### remove padding values
  dat <- dat[dat$doy != 0, ]

  ### we need to adjust the doy values in dat to match the observations
  ### the doy range is 1 to 365 in dat, but 1 to 4*365 (cumulative across the years) in observations
  ### we need to go through dat$doy and each time the doy does NOT increase, we need to add 365 until it does
  for (i in 2:length(dat$doy)) {
    while (dat$doy[i] <= dat$doy[i-1]) {
      dat$doy[i] <- dat$doy[i] + 365
    }
  }

  ### read the corresponding observations file (mainly to retrieve the original date column)
  observations <- read.table(paste0(base_path, "/", substr(preds[iter], 1, nchar(preds[iter])-4), ".csv"), 
                    header = TRUE, sep = ",")
  ### format the date column correctly
  observations$date <- as.character(as.Date(observations$date, format = "%Y-%m-%d"))
  ### select only the necessary columns
  observations <- observations[, colnames(observations) %in% c("DOY", "date")]
  ### rename the DOY column to match the dat dataframe
  colnames(observations)[which(colnames(observations) == "DOY")] <- "doy"
  ### join the observations (original dates) with the dat dataframe by doy
  dat <- left_join(dat, observations, by = "doy")

  if (infested == TRUE){
    ### remove the rows with dat$date later than 2021-12-31, 
    ### because the time frame for validation is only until 2021-12-31
    dat <- dat[dat$date <= "2021-12-31", ]
  } else {
    ### remove the rows with dat$date later than 2020-12-31, 
    ### because the time frame for undisturbed test dataset is only until 2020-12-31
    dat <- dat[dat$date <= "2020-12-31", ]
  }
  
  ### retrieve metadata information for the current prediction
  ### this is done by matching the plotID in the metadata with the plotID in the preds file
  metadata <- meta[which(meta$plotID == substr(preds[iter], 1, nchar(preds[iter])-4)), ]

  ### loop through the bands defined in the main script
  ### and calculate + plot the anomaly score for each band
  for (band in bands) {
    ### if MSE was chosen (default), calculate the anomaly score as squared error
    ### if relative absolute error was chosen (MSE <- FALSE), calculate the anomaly score as relative absolute error
    if (mse == TRUE){
      dat[[paste0("anomaly_score_", band)]] <- (dat[[band]] - dat[[paste0(band, "_pred")]])^2
    } else { # relative absolute error
      abs_diff <- abs(dat[[band]] - dat[[paste0(band, "_pred")]])
      abs_observed <- abs(dat[[band]])
      ### define a small epsilon to avoid division by zero
      epsilon <- 1e-8
      dat[[paste0("anomaly_score_", band)]] <- abs_diff / (abs_observed + epsilon)
    }

    ##### start preparing the plots #####

    ### calculate primary y-axis range
    min_primary <- min(c(dat[[band]], dat[[paste0(band, "_pred")]]))
    max_primary <- max(c(dat[[band]], dat[[paste0(band, "_pred")]]))

    ### define anomaly score range
    min_anomaly_score <- 0.0
    max_anomaly_score <- (max(dat[[paste0("anomaly_score_", band)]])) + (0.1 * (max(dat[[paste0("anomaly_score_", band)]])))

    ### create a named vector for colors
    color_values <- setNames(c("blue", "red"), c(band, paste0(band, "_pred")))

    ### upper panel: band and band_pred line plot
    p1 <- ggplot(dat, aes(x = as.POSIXct(date, format = "%Y-%m-%d"))) +
      # add the observed band line
      geom_line(aes_string(y = band, color = shQuote(band)), linewidth = 0.6) +
      # add the predicted band line
      geom_line(aes_string(y = paste0(band, "_pred"), color = shQuote(paste0(band, "_pred"))), linewidth = 0.6) +
      # assign colors
      scale_color_manual(values = color_values) + 
      # format x and y axes
      scale_x_datetime(date_breaks = "2 months", date_labels = "%Y-%m") + 
      scale_y_continuous(
        name = paste0("Normalized ", band),
        limits = c(min_primary - 0.1 * (max_primary - min_primary), max_primary + 0.1 * (max_primary - min_primary))  # Adjust primary y-axis limits
      ) +
      # set "Date" as x-axis label
      labs(x = "Date", color = "") +  
      # assign a suitable theme
      theme_bw() + 
      # some additional theme settings like font size, angle of x-axis labels, etc.
      theme(
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), 
        text = element_text(size = 18),  # Increase font size for all text
        axis.title = element_text(size = 18),  # Increase font size for axis titles
        axis.text = element_text(size = 16),  # Increase font size for axis labels
        legend.text = element_text(size = 18),  # Increase font size for legend text
        legend.title = element_text(size = 20),  # Increase font size for legend title
        plot.margin = unit(c(1, 1, 1, 1), "cm")  # Adjust plot margins
      )

    ### lower panel: anomaly score plot
    p2 <- ggplot(dat, aes(x = as.POSIXct(date, format = "%Y-%m-%d"))) +
      # add the anomaly score line
      geom_line(aes_string(y = paste0("anomaly_score_", band)), color = "black", linewidth = 1) +
      # add shaded area for anomaly score
      geom_ribbon(aes_string(ymin = 0, ymax = paste0("anomaly_score_", band)), fill = "gray80", alpha = 0.5) +
      # optional: add horizontal dashed line at anomaly score
      # if desired, uncomment the next line and define a suitable yintercept value (the threshold value)
      # geom_hline(yintercept = 1.138, linetype = "dashed", color = "gray40", linewidth = 0.5) + 
      # format x and y axes
      scale_x_datetime(date_breaks = "2 months", date_labels = "%Y-%m") + 
      scale_y_continuous(
        name = "Anomaly Score (Squared Error)",
        limits = c(min_anomaly_score, max_anomaly_score)  # Adjust anomaly score y-axis limits
      ) +
      # set "Date" as x-axis label
      labs(x = "Date") +
      # assign a suitable theme
      theme_bw() +
      # some additional theme settings like font size, angle of x-axis labels, etc.
      theme(
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), 
        text = element_text(size = 18),  # Increase font size for all text
        axis.title = element_text(size = 18),  # Increase font size for axis titles
        axis.text = element_text(size = 16),  # Increase font size for axis labels
        legend.text = element_text(size = 18),  # Increase font size for legend text
        legend.title = element_text(size = 20),  # Increase font size for legend title
        plot.margin = unit(c(1, 1, 1, 1), "cm")  # Adjust plot margins
      )

    ### combine the two panels using patchwork
    combined_plot <- p1 / p2

    ### save the plot
    if (mse == TRUE){
      ggsave(paste0(base_path, '/models/', model_dir_name, '/plots/single_plots/', substr(preds[iter], 1, nchar(preds[iter])-4), "_", metadata$DMG, "_", band, "_MSE.pdf"), plot = combined_plot, width = 12, height = 12)
    } else {
      ggsave(paste0(base_path, '/models/', model_dir_name, '/plots/single_plots/', substr(preds[iter], 1, nchar(preds[iter])-4), "_", metadata$DMG, "_", band, "_relabserror.pdf"), plot = combined_plot, width = 12, height = 12)
    }

    ### end of "for band in bands"
  }

### end of parallel function
}


### define base variables
# base path to data/result directory, which contains the model folder (see MODEL_DIR_NAME below) and the meta folder
# (which in turn contains the metadata.csv file)
# this is the same as "root_path" in the training script
# if you run 'Rscript get_results/visualize_time_series.r' from the terminal in the main project folder, you likely don't have to change anything
BASE_PATH <- paste0(getwd(), '/dataset')
# name of the model (likely determined by the model training/test run already - just copy-paste the directory name here)
# if you trained the default model as in the tutorial, you likely don't have to change anything here
MODEL_DIR_NAME <- 'anomaly_detection_RsTs_TimesNet_RSAD_RsTs_sl200_dm4_el2_df4_eblearnedsincos_test_run_seed123_0'
### number of CPUs used for parallel processing
### default is 1, but you can set it to the number of CPUs you want to use
NUM_WORKERS <- 1

# mean squared error (default) or relative absolute error?
MSE <- TRUE # TRUE for MSE, FALSE for relative absolute error
### define bands to be checked
### order of the bands must be the same as in training script!
# default value: just CRSWIR
bands <- c('CRSWIR')
# in case you want to use all bands, uncomment the following line
# bands <- c('BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2')

##### this is the start of the data preparation and plotting
##### firstly, plot the results of the undisturbed test dataset #####

### get a list of files with predictions
preds <- list.files(paste0(BASE_PATH, '/models/', MODEL_DIR_NAME, '/preds'), pattern = ".txt")

### read the metadata file
meta <- read.table(paste0(BASE_PATH, "/meta/metadata.csv"), header = TRUE, sep = ",")
### subset metadata file to the undisturbed test dataset
meta <- meta[(meta$dataset %in% c("may") & meta$DMG == 0), ]

### subset preds by meta$plotID (because we will iterate through preds, so these must match the target files)
preds <- preds[substr(preds, 1, nchar(preds)-4) %in% meta$plotID]

### create the directory for the plots if it does not exist
dir.create(paste0(BASE_PATH, '/models/', MODEL_DIR_NAME, '/plots/single_plots'), showWarnings = FALSE, recursive = TRUE)


### execute the parallel function using future_map
### first, we need to start a "plan" (parallel backend) for the future_map function
### note that this might only work on a Linux system,
### Windows users might have to use plan("multisession", workers = NUM_WORKERS) instead
plan(multicore, workers = NUM_WORKERS)

print("starting parallel task")
### the future_map function will iterate through the preds list
### and call the plot_example_time_series_and_anomaly_score function in parallel to speed up the plotting process
### it passes all the necessary parameters to the function (e.g. preds, base_path, ...)
future_map(1:length(preds), plot_example_time_series_and_anomaly_score,
           preds = preds, base_path = BASE_PATH, model_dir_name = MODEL_DIR_NAME, mse = MSE, meta = meta, bands = bands, infested = FALSE)
print("parallel task finished")

plan("sequential")
### end of plotting the undisturbed test dataset results

##### now plot the results of the validation dataset #####

### get a list of files with predictions
preds <- list.files(paste0(BASE_PATH, '/models/', MODEL_DIR_NAME, '/preds'), pattern = ".txt")

### read the metadata file
meta <- read.table(paste0(BASE_PATH, "/meta/metadata.csv"), header = TRUE, sep = ",")
### subset metadata file to the validation dataset
meta <- meta[(meta$dataset %in% c("may") & meta$DMG == 1), ] # , "may_val_shapes"

### subset preds by meta$plotID
preds <- preds[substr(preds, 1, nchar(preds)-4) %in% meta$plotID]


### execute the parallel function using future_map
### first, we need to start a "plan" (parallel backend) for the future_map function
### note that this might only work on a Linux system,
### Windows users might have to use plan("multisession", workers = NUM_WORKERS) instead
plan(multicore, workers = NUM_WORKERS)

print("starting parallel task")
### the future_map function will iterate through the preds list
### and call the plot_example_time_series_and_anomaly_score function in parallel to speed up the plotting process
### it passes all the necessary parameters to the function (e.g. preds, base_path, ...)
future_map(1:length(preds), plot_example_time_series_and_anomaly_score,
           preds = preds, base_path = BASE_PATH, model_dir_name = MODEL_DIR_NAME, mse = MSE, meta = meta, bands = bands, infested = TRUE)
print("parallel task finished")

plan("sequential")
### end of plotting the validation dataset results

print("All plots have been created successfully! Exiting...")
##### end of the script #####