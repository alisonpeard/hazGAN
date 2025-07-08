"Identify independent storm events.

Includes event metadata such as return period and frequency. Only fit ECDF and 
GPD to 1940-2022 data. Keep 2021 as a holdout test set.

Files:
------
- input:
  - data_1941_2022.nc

- output:
  - storms.parquet
  - metadata.parquet
  - medians.csv

LAST RUN: 17-02-2025
"
#%%######### START #############################################################
# Clear environment
rm(list = ls())

library(arrow)
library(lubridate)
library(dplyr)
require(ggplot2)
library(CFtime)
library(tidync)
source("utils/utils.R")
source("utils/settings.R")

# set up env (depends if running or sourcing script)
try(setwd(getSrcDirectory(function(){})[1]))
try(setwd(dirname(rstudioapi::getActiveDocumentContext()$path)))

readRenviron("../../.env")

WD         <- Sys.getenv("ERA5DIR")

#%%######### LOAD AND STANDARDISE DATA #########################################
print("Loading and standardising data...")
start <- Sys.time()
src <- tidync(paste0(WD, "/", FILENAME))

# subset to mini dataset if it's a dry run
if(DRYRUN) {
  lats <- seq(10, 25, length.out=RES[1])
  lons <- seq(80, 95, length.out=RES[2])

  src <- src %>%
    hyper_filter(
      lon = lon <= lons[NDRYRUN],
      lat = lat <= lats[NDRYRUN]
    )
}

daily  <- src %>% hyper_tibble(force = TRUE)
coords <- src %>% activate("grid") %>% hyper_tibble(force = TRUE)
daily  <- left_join(daily, coords, by = c("lon", "lat"))

rm(coords)

daily      <- daily[, c("grid", "time", "u10", "msl", "tp")]
daily$msl  <- -daily$msl # negate pressure so maximizing all vars
daily$time <- as.Date(STARTDATE) + days(daily$time)
daily$grid <- as.integer(daily$grid)

std_u10 <- standardise_by_month(daily, "u10")
std_msl <- standardise_by_month(daily, "msl")
std_tp  <- standardise_by_month(daily, "tp")

#%%######## EXTRACT STORMS #####################################################
print("Extracting storms...")
metadata   <- storm_extractor(daily, "u10", RFUNC)

########### SAVE RESULTS #######################################################
print("Finished storm extraction. Saving...")
write.csv(medians, paste0(WD, "/", "medians.csv"), row.names = FALSE)
write_parquet(metadata, paste0(WD, "/", "storms_metadata.parquet"))
write_parquet(daily, paste0(WD, "/", "daily.parquet"))

########### END ################################################################
