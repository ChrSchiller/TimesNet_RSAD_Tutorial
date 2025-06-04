##### This script takes the predictions from the model and quantifies the results
##### first, it uses the validation errors to automatically determine a threshold
##### this decision threshold is used to detect anomalies in the time series and their exact date of occurrence
##### afterwards, the confusion matrix is computed, and performance metrics are calculated
##### finally, the results are mapped and saved to disk

### specify your packages
my_packages <- c('terra', 'ggplot2', 'tidyr', 'scales', 'ggthemes', 'future', 'stringr', 'prettymapr',
                 'future.apply', 'furrr', 'purrr', 'dplyr', 'lubridate', 'sf', 'ggspatial', 'viridis')
### extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , 'Package'])]
### install not installed packages
if(length(not_installed)) install.packages(not_installed)


### imports
require(dplyr)
library(future)
library(future.apply)
library(furrr)
library(purrr)
library(caret)
require(lubridate)
library(magick)
library(ggplot2)
require(stringr)
require(terra)
require(sf)
require(ggspatial)
require(prettymapr)
require(scales)
require(viridis)

### this function reads each file containing the predictions
### and calculates the breakpoints throughout the time series
### it returns a list with the results
### the function works on both undisturbed and infested datasets/files
get_results <- function(file_iter, meta, threshold, repetitions, drop_winter, band, healthy, base_path, model_dir_name, bands){

  ### read dataset containing the predictions
  dat <- read.table(paste0(base_path, '/models/', model_dir_name, '/preds/', meta$plotID[file_iter], ".txt"), 
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
  observations <- read.table(paste0(base_path, "/", meta$plotID[file_iter], ".csv"), 
                    header = TRUE, sep = ",")
  ### format the date column correctly
  observations$date <- as.character(as.Date(observations$date, format = "%Y-%m-%d"))
  ### select only the necessary columns
  observations <- observations[, colnames(observations) %in% c("DOY", "date")]
  ### rename the DOY column to match the dat dataframe
  colnames(observations)[which(colnames(observations) == "DOY")] <- "doy"
  ### join the observations (original dates) with the dat dataframe by doy
  dat <- left_join(dat, observations, by = "doy")
  
  ### remove duplicates (should not exist; just a safety measure)
  dat <- dat[!duplicated(dat$doy, keep = 'first'), ]

  ### calculate anomaly score for "band" for each observation
  ### anomaly score is the mean squared error
  dat$anomaly_score <- (dat[[paste0(band, "_pred")]] - dat[[band]])^2

  ### format the date column as date
  dat$date <- as.Date(dat$date, format = "%Y-%m-%d")

  ### we start all undisturbed time series from 2018-01-01
  ### and all infested time series from 2019-01-01
  ### this is because we only have information about the undisturbed status until 2020-12-31
  ### and want to investigate three years of monitoring
  ### for infestations, we know that the disturbance happened in 2021
  ### and again, we want three years of monitoring
  if (healthy == TRUE){
  dat <- dat[dat$date >= as.Date("20180101", format = "%Y%m%d"), ]
  } else {
    dat <- dat[dat$date >= as.Date("20190101", format = "%Y%m%d"), ]
  }
  
  ### if desired, remove winter observations from analysis
  if (drop_winter == TRUE){
    ### filter out observations in "winter" (arbitrarily defined as Nov 15th until March 15th)
    dat <- dat %>%
      filter(!(format(date, "%m-%d") >= "11-15" | format(date, "%m-%d") <= "03-15"))
  }
  
  ### subset the dataframe to a specific end date
  if (healthy == TRUE){
    ### for undisturbed reference data, the last observation date (confirming health status) is 2021-06-14
    ### but we want to exclude potential mistakes because of undetected early infestations in this dataset
    ### so we only consider the time until 2021-01-01
    dat <- dat %>%
      filter(as.character(date) <= "2021-01-01")
  } else {
    ### for the infested reference data, the infestation observation took place in summer 2021
    ### so we end the detection period at the end of 2021
    dat <- dat %>%
      filter(as.character(date) <= "2022-01-01")
  }
  
  ### we need at least three observations to continue
  ### (this is just a precaution to avoid errors in the for loop)
  if (nrow(dat) > 3){
    
    ### get anomalies: "repetitions" consecutive observations >= threshold
    # define some variables as starting points
    count <- 0
    first_index <- NULL

    ### intialize list of breakpoints
    breakpoint_list <- c()

    ### check if the first two observations meet the condition
    if (all(dat$anomaly_score[1:repetitions] >= threshold)) {
        count <- count + 1
        first_index <- repetitions
        # last_index <- repetitions

        ### append date of detection to breakpoint_list and remove the "-" symbols to get a date in YYYYMMDD format
        breakpoint_list <- c(breakpoint_list, str_replace_all(as.character(dat$date[repetitions]), "-", ""))
    }

    ### loop through the time series to check for following breakpoints
    for (i in 1:(nrow(dat) - repetitions)) {

      ### check the condition: breakpoint detected?
      if (all(dat$anomaly_score[i + 1:repetitions] >= threshold)) {

        ### add detection date to breakpoint_list
        breakpoint_list <- c(breakpoint_list, str_replace_all(as.character(dat$date[i + repetitions]), "-", ""))

        ### increase count
        count <- count + 1
        
        ### update first_index if it's the first time the condition is met
        if (is.null(first_index)) {
            first_index <- i + repetitions
        }
        
      ### end of if (all(dat$anomaly_score[i + 1:repetitions] >= threshold)) {
      }

    ### end of for (i in 1:(nrow(dat) - repetitions)) {
    }

    ### damage was predicted in case count > 0 -> assign 1, else assign 0
    pred_dmg <- as.numeric(count > 0)

    ### prepare breakpoints: we want to concatenate all the 
    ### entries in breakpoint_list to a single string
    ### each entry separated by a comma
    breakpoints <- paste(breakpoint_list, collapse = ",")
          
      ### extract and return values
      return(
        list(
          plotID = meta$plotID[file_iter],
          pred_dmg = pred_dmg,
          no_alarms = count,
          breakpoints = breakpoints,
          ### first_tod: if first_index not NULL, then use as.character(dat$date[first_index]), else NA
          first_tod = ifelse(!is.null(first_index), as.character(dat$date[first_index]), NA)
        )
      )
  
    ### end of "if (nrow(dat) > 3){"
  }

  ### end of get_results function
}


### define base variables
# base path to data/result directory, which contains the model folder (see MODEL_DIR_NAME below) and the meta folder
# (which in turn contains the metadata.csv file)
# this is the same as "root_path" in the training script
# if you run 'Rscript get_results/visualize_time_series.r' from the terminal in the main project folder, you likely don't have to change anything
BASE_PATH <- paste0(getwd(), '/dataset')
# name of the model (likely determined by the model training/test run already - just copy-paste the directory name here)
# if you trained the default model as in the tutorial, you likely don't have to change anything herre
MODEL_DIR_NAME <- 'anomaly_detection_RsTs_TimesNet_RSAD_RsTs_sl200_dm4_el2_df4_eblearnedsincos_test_run_seed123_0'
### number of CPUs used for parallel processing
### default is 1, but you can set it to the number of CPUs you want to use
NUM_WORKERS <- 1

### define which band to analyze
BAND <- "CRSWIR"
BANDS <- c("CRSWIR")
# BANDS might be a larger list, e.g. in case of all 10 bands, but we want to look at only one band at a time
# BANDS <- c("BLU", "GRN", "RED", "RE1", "RE2", "RE3", "BNR", "NIR", "SW1", "SW2") # list of bands in the time series
### define threshold factor as in Schiller et al. (2025)
FACTOR <- 1 # default: 1
### define repetitions for the anomaly detection
### -> how many consecutive observations need to be above the threshold to raise an alarm?
REPETITIONS <- 2 # default: 2
### drop winter observations? (15 Nov until 15 Mar)
DROP_WINTER <- TRUE # default: TRUE
### path to the geopackage containing the validation pixels and polygons
VAL_PIXELS_PATH <- paste0(BASE_PATH, "/meta/val_pixels.gpkg")
VAL_POLYS_PATH <- paste0(BASE_PATH, "/meta/val_polys.gpkg")

##### automated thresholding procedure for the detection
### get the validation errors
vali_energy <- read.table(paste0(BASE_PATH, "/models/", MODEL_DIR_NAME, "/single_val_losses_each_band.txt"), header = FALSE, sep = ",")

### filter the correct/desired band
vali_enery <- vali_energy[which(BANDS == BAND) + seq(0, nrow(vali_energy) - which(BANDS == BAND), by = length(BANDS)), ]

### remove all padding values (== 0.0)
vali_energy <- vali_energy[vali_energy != 0.0]

### define some statistics to determine the threshold
mean_energy <- mean(vali_energy)
sd_energy <- sd(vali_energy) # too much influence of outliers, so we don't use it
q01_energy <- quantile(vali_energy, 0.01)
q99_energy <- quantile(vali_energy, 0.99)
iqr_energy <- q99_energy - q01_energy

### define the threshold with the mean and the interquantile range
THRESHOLD <- mean_energy + FACTOR * iqr_energy
print(paste0("Threshold for raising an alarm: ", round(THRESHOLD, 3)))


##### prepare the datasets for analysis #####

##### prepare undisturbed meta
### get a list of files with predictions
preds <- list.files(paste0(BASE_PATH, '/models/', MODEL_DIR_NAME, '/preds'), pattern = ".txt")

### read the metadata file
meta <- read.table(paste0(BASE_PATH, "/meta/metadata.csv"), header = TRUE, sep = ",")
### subset metadata file to the undisturbed test dataset
meta_undisturbed <- meta[(meta$dataset %in% c("may") & meta$DMG == 0), ]

### subset preds by meta$plotID (because we will iterate through preds, so these must match the target files)
preds_undisturbed <- preds[substr(preds, 1, nchar(preds)-4) %in% meta_undisturbed$plotID]


##### prepare infested meta
### subset metadata file to the validation dataset
meta_dmg <- meta[(meta$dataset %in% c("may") & meta$DMG == 1), ] # , "may_val_shapes"

### subset preds by meta$plotID
preds_dmg <- preds[substr(preds, 1, nchar(preds)-4) %in% meta_dmg$plotID]


##### run through the two datasets (undisturbed and infested) #####

print(paste0("FACTOR: ", FACTOR, ", THRESHOLD: ", round(THRESHOLD, 4), ", REPETITIONS: ", REPETITIONS, ", DROP_WINTER: ", DROP_WINTER, 
              ", BAND: ", BAND))


##### undisturbed dataset #####

### execute the parallel function using future_map
### first, we need to start a "plan" (parallel backend) for the future_map function
### note that this might only work on a Linux system,
### Windows users might have to use plan("multisession", workers = NUM_WORKERS) instead
plan(multicore, workers = NUM_WORKERS) # this seems to do something in Terminal, but not in RStudio

print("starting parallel task")
HEALTHY <- TRUE
### the future_map function will iterate through the metafile's observations
### and call the get_results function in parallel to speed up the plotting process
### it passes all the necessary parameters to the function (e.g. meta_undisturbed, THRESHOLD, ...)
### the function returns a list of results for each observation
results <- future_map(1:nrow(meta_undisturbed), function(main_iter){get_results(
  main_iter, meta_undisturbed, THRESHOLD, REPETITIONS, DROP_WINTER, BAND, HEALTHY, BASE_PATH, MODEL_DIR_NAME, BANDS)})
print("parallel task finished")

### stop the future plan
plan('sequential')

### convert to dataframe
res <- as.data.frame(bind_rows(results))

### have a look at detected breakpoints
print(res$breakpoints[res$breakpoints != ""])

### join the results to the meta file
undisturbed <- left_join(meta_undisturbed, res, by = c("plotID"))

print("Show some results: ")
print(table(res$no_alarms, exclude = FALSE))
print(table(res$pred_dmg, exclude = FALSE))

### save to disk as intermediate save
tmp_folder <- paste0(BASE_PATH, "/models/", MODEL_DIR_NAME, "/results/", "threshold_", gsub("\\.", "_", as.character(round(THRESHOLD, 4))), 
                    "_rep_", as.character(REPETITIONS), "_dw_", tolower(as.character(DROP_WINTER)), "/tmp")
dir.create(tmp_folder, recursive = TRUE, showWarnings = FALSE)

### do some data polishing, because sometimes the date format is not preserved properly (convert to character)
undisturbed$first_tod <- as.character(undisturbed$first_tod)
undisturbed$breakpoints <- as.character(undisturbed$breakpoints)
undisturbed$breakpoints[undisturbed$breakpoints == ""] <- NA 
write.table(undisturbed, paste0(tmp_folder, "/dataset_undisturbed.csv"), sep = ";", row.names = FALSE)

##### infested dataset #####

### the future_map function will iterate through the metafile's observations
### and call the get_results function in parallel to speed up the plotting process
### it passes all the necessary parameters to the function (e.g. meta_dmg, THRESHOLD, ...)
### the function returns a list of results for each observation
plan(multicore, workers = NUM_WORKERS) # this seems to do something in Terminal, but not in RStudio

print("starting parallel task")
HEALTHY <- FALSE
results <- future_map(1:nrow(meta_dmg), function(main_iter){get_results(
  main_iter, meta_dmg, THRESHOLD, REPETITIONS, DROP_WINTER, BAND, HEALTHY, BASE_PATH, MODEL_DIR_NAME, BANDS)})
print("parallel task finished")
### note that one of the "no_alarms" is the correct alarm in case of DMG == 1!!!

# stop the future plan
plan('sequential')

### convert to dataframe
res <- as.data.frame(bind_rows(results))

### check some results
table(res$pred_dmg, exclude = FALSE)
res$breakpoints[res$breakpoints != ""]
table(res$no_alarms)

### join the results to the meta file
infested <- left_join(meta_dmg, res, by = c("plotID"))

print("Show some results: ")
print(table(infested$pred_dmg, exclude = FALSE))
print(table(infested$no_alarms, exclude = FALSE))
res[infested$no_alarms > 0, ]

### save to disk as preliminary save
infested$first_tod <- as.character(infested$first_tod)
infested$breakpoints <- as.character(infested$breakpoints)
infested$breakpoints[infested$breakpoints == ""] <- NA
write.table(infested, paste0(tmp_folder, "/dataset_infested.csv"), sep = ";", row.names = FALSE)


#############################################
##### concatenation and result analysis #####
#############################################

##### concatenate and analyze: accuracy + average early warning time (for correct predictions)
train_hlthy <- read.table(paste0(tmp_folder, "/dataset_undisturbed.csv"), sep = ";", header = TRUE)
val_dmg <- read.table(paste0(tmp_folder, "/dataset_infested.csv"), sep = ";", header = TRUE)

### bin the two datasets together
dat <- rbind(train_hlthy, val_dmg)

### compute the results for confusion matrix
tp <- sum((dat$pred_dmg) == 1 & (dat$DMG) == 1)
fp <- sum((dat$pred_dmg) == 1 & (dat$DMG) == 0)
tn <- sum(dat$DMG == 0 & dat$pred_dmg == 0)
fn <- sum(dat$DMG == 1 & dat$pred_dmg == 0)

### compute precision, recall, and f1-score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (tp + tn) / (tp + fp + tn + fn)
balanced_accuracy = (recall + tn / (tn + fp)) / 2

### print the results
print(paste0("Precision: ", round(precision, 5)))
print(paste0("Recall: ", round(recall, 5)))
print(paste0("F1: ", round(f1, 5)))
print(paste0("Balanced accuracy: ", round(balanced_accuracy, 5)))
print(paste0("Overall accuracy: ", round(accuracy, 5)))
print(paste0("True positives: ", tp))
print(paste0("False positives: ", fp))
print(paste0("True negatives: ", tn))
print(paste0("False negatives: ", fn))

### write results to disk
file_path <- paste0(BASE_PATH, "/models/", MODEL_DIR_NAME, "/results/", "threshold_", gsub("\\.", "_", as.character(round(THRESHOLD, 4))), 
                    "_rep_", as.character(REPETITIONS), "_dw_", tolower(as.character(DROP_WINTER)), "/results.txt")

### open the file in write mode with "append = TRUE" to avoid overwriting existing content
sink(file = file_path, append = TRUE)

### print all results computed above
print("Results for the current setup:")
print(paste0("FACTOR: ", FACTOR, ", THRESHOLD: ", round(THRESHOLD, 4), ", REPETITIONS: ", REPETITIONS, ", DROP_WINTER: ", DROP_WINTER, 
              ", BAND: ", BAND))
print(paste0("Precision: ", round(precision, 5)))
print(paste0("Recall: ", round(recall, 5)))
print(paste0("F1: ", round(f1, 5)))
print(paste0("Balanced accuracy: ", round(balanced_accuracy, 5)))
print(paste0("Overall accuracy: ", round(accuracy, 5)))
print(paste0("True positives: ", tp))
print(paste0("False positives: ", fp))
print(paste0("True negatives: ", tn))
print(paste0("False negatives: ", fn))

print("Distribution of number of alarms for infested time series:")
print(table(dat$no_alarms[dat$DMG == 1], exclude = FALSE))
print("Distribution of number of alarms for undisturbed time series:")
print(table(dat$no_alarms[dat$DMG == 0], exclude = FALSE))

### close the sink connection
sink()

### in this section, we create maps to get an overview of correct and false predictions
### we need to left_join the results (dat) to the geopackages 
val_pixels <- vect(VAL_PIXELS_PATH)

### merge the information from dat to the train/val pixels spatvector
val_pixels <- terra::merge(val_pixels, dat, by = "plotID")

### save results to disk
writeVector(val_pixels, paste0(BASE_PATH, "/models/", MODEL_DIR_NAME, "/results/", 
                                "threshold_", gsub("\\.", "_", as.character(round(THRESHOLD, 4))), 
                                "_rep_", as.character(REPETITIONS), "_dw_", tolower(as.character(DROP_WINTER)), "/val_pixels_results.gpkg"), overwrite = TRUE)

### read the polygons
val_polys <- vect(VAL_POLYS_PATH)
val_polys <- terra::project(val_polys, crs(val_pixels))  # ensure both have the same CRS

### define nice colors for healthy and infested
healthy_color <- "#56B4E9"  # light blue
infested_color <- "#E69F00"  # orange

### create maps directory if it does not exist
dir.create(paste0(BASE_PATH, "/models/", MODEL_DIR_NAME, "/results/", "threshold_", gsub("\\.", "_", as.character(round(THRESHOLD, 4))), 
                    "_rep_", as.character(REPETITIONS), "_dw_", tolower(as.character(DROP_WINTER)), "/maps"), recursive = TRUE, showWarnings = FALSE)


### create the plot: pred_dmg provides healthy (== 0) and infested (== 1) information
### note that we have a SpatVector, so we need a spatial version of ggplot
plot <- ggplot() +
  ggspatial::annotation_map_tile(type = "osm") +  # Add OSM background
  # Add polygons in the background with alpha = 0.5 and same color scheme
  geom_sf(
    data = st_as_sf(val_polys[val_polys$DMG == 0, ]),
    fill = healthy_color, color = "gray80", alpha = 0.5
  ) +
  geom_sf(
    data = st_as_sf(val_polys[val_polys$DMG == 1, ]),
    fill = infested_color, color = "gray80", alpha = 0.5
  ) +
  # Add main pixels on top
  geom_sf(data = st_as_sf(val_pixels[val_pixels$pred_dmg == 0, ]), aes(fill = "Healthy"), color = "black") +
  geom_sf(data = st_as_sf(val_pixels[val_pixels$pred_dmg == 1, ]), aes(fill = "Infested"), color = "black") +
  scale_fill_manual(values = c("Healthy" = healthy_color, "Infested" = infested_color), name = "Prediction", drop = FALSE) +
  coord_sf(crs = st_crs(25832)) +  # set the coordinate reference system to EPSG:25832
  theme_bw() +
  theme(
    panel.grid = element_blank(),  # remove grid lines
    legend.position = "right",
    legend.title = element_text(size = 20),  # increase legend title font size
    legend.text = element_text(size = 18),   # increase legend text font size
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank()
  ) + 
  annotation_north_arrow(location = "br", which_north = "true", width = unit(3, "cm"), height = unit(3, "cm"),
                        style = north_arrow_fancy_orienteering)  # Add north arrow

### write the plot to disk
ggsave(paste0(BASE_PATH, "/models/", MODEL_DIR_NAME, "/results/", 
                  "threshold_", gsub("\\.", "_", as.character(round(THRESHOLD, 4))), 
                  "_rep_", as.character(REPETITIONS), "_dw_", tolower(as.character(DROP_WINTER)), "/maps/result_map.pdf"), plot, width = 12, height = 8)



### create a map of the first time of detection using a color gadient to check detection delay
### revelant is the first_tod column, which contains the first date of detection
### the color fill of the pixels will show the detection dates derived from first_tod column
### the color gradient for that will be from viridis palette, which is perceptually uniform
### and the stroke around the pixels will show the DMG (ground truth undisturbed or infested)
### not/never detected (first_tod == NA) will show up with fill color gray

### prepare fill column: NA for never detected, numeric for detected
val_pixels$first_tod <- as.Date(val_pixels$first_tod, format = "%Y-%m-%d")
val_pixels$fill_detection <- as.numeric(val_pixels$first_tod)
val_pixels$fill_detection[is.na(val_pixels$first_tod)] <- NA  # keep NA for never detected

### define breaks and labels for the legend (e.g., 5 evenly spaced dates)
date_range <- range(val_pixels$first_tod, na.rm = TRUE)
breaks <- seq(date_range[1], date_range[2], length.out = 5)
breaks_num <- as.numeric(breaks)
labels <- format(breaks, "%Y-%m-%d")

### create the plot
plot_first_tod <- ggplot() +
  ### add the OpenStreetMap background
  ggspatial::annotation_map_tile(type = "osm") +
  ### add the polygons in the background
  geom_sf(
    data = st_as_sf(val_polys[val_polys$DMG == 0, ]),
    fill = healthy_color, color = "gray80", alpha = 0.5
  ) +
  geom_sf(
    data = st_as_sf(val_polys[val_polys$DMG == 1, ]),
    fill = infested_color, color = "gray80", alpha = 0.5
  ) +
  ### add the main pixels with the first detection dates
  geom_sf(
    data = st_as_sf(val_pixels[val_pixels$DMG == 0, ]),
    aes(fill = fill_detection),
    color = "black", linetype = "solid"
  ) +
  geom_sf(
    data = st_as_sf(val_pixels[val_pixels$DMG == 1, ]),
    aes(fill = fill_detection),
    color = "black", linetype = "dashed"
  ) +
  ### add the color gradient for the first detection dates
  scale_fill_viridis_c(
    na.value = "gray",
    name = "First detection",
    breaks = breaks_num,
    labels = labels
  ) +
  ### set the coordinate reference system to EPSG:25832
  coord_sf(crs = st_crs(25832)) +
  ### change some display settings
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 18),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank()
  ) +
  annotation_north_arrow(location = "br", which_north = "true", width = unit(3, "cm"), height = unit(3, "cm"),
                        style = north_arrow_fancy_orienteering)

### save to disk
ggsave(paste0(BASE_PATH, "/models/", MODEL_DIR_NAME, "/results/", 
                  "threshold_", gsub("\\.", "_", as.character(round(THRESHOLD, 4))), 
                  "_rep_", as.character(REPETITIONS), "_dw_", tolower(as.character(DROP_WINTER)), "/maps/first_detection_date_map.pdf"), 
       plot_first_tod, width = 12, height = 8)
