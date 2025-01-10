"Identify events and fit to marginals.

Includes event metadata such as return period and frequency. Only fit ECDF and 
GPD to 1940-2021 data. Keep 2022 as a holdout test set.

Best run in VS code.

Files:
-----
- input:
  - data_1940_2022.nc

- output:
  - storms.parquet
  - metadata.parquet
  - medians.csv

To do:
------
- train-test split before ECDF & GPD fit (nearly there)
- convert to Python!

LAST RUN: 08-12-2024
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

# set up env (depends if running or sourcing script)
try(setwd(getSrcDirectory(function(){})[1]))
try(setwd(dirname(rstudioapi::getActiveDocumentContext()$path)))
source("utils.R")
readRenviron("../../.env")

FILENAME   <- "data_1941_2022.nc"     # nolint
RES        <- c(64, 64)               # nolint
WD         <- Sys.getenv("TRAINDIR")  # nolint
WD         <- paste0(WD, "/", res2str(RES))
RFUNC      <- max                     # nolint, https://doi.org/10.1111/rssb.12498
TEST.YEARS <- c(2022)                 # nolint, exclude from ecdf + gpd fitting
VISUALS    <- TRUE                    # nolint
Q          <- 0.8                     # nolint

#%%######### LOAD AND STANDARDISE DATA #########################################
print("Loading and standardising data...")
src <- tidync(paste0(WD, "/", FILENAME))
daily <- src %>% hyper_tibble(force = TRUE)
coords <- src %>% activate("grid") %>% hyper_tibble(force = TRUE)
daily <- left_join(daily, coords, by = c("lon", "lat"))
rm(coords)

daily$msl <- -daily$msl # negate pressure so maximizing all vars
daily <- daily[, c("grid", "time", "u10", "msl", "tp")]
daily$time <- as.Date("1941-01-01") + days(daily$time)

medians <- monthly_medians(daily, "u10")
medians$mslp <- monthly_medians(daily, "msl")$msl
medians$tp <- monthly_medians(daily, "tp")$tp

daily$u10 <- standardise_by_month(daily, "u10")
daily$msl <- standardise_by_month(daily, "msl")
daily$tp <- standardise_by_month(daily, "tp")

#%%######## EXTRACT AND TRANSFORM STORMS #######################################
print("Extracting storms...")
metadata <- storm_extractor(daily, "u10", RFUNC)

# fit to marginal data
print("Tranforming fields...")
storms_wind <- gpd_transformer(daily, metadata, "u10", Q)
storms_mslp <- gpd_transformer(daily, metadata, "msl", Q)
storms_tp   <- gpd_transformer(daily, metadata, "tp", Q)

print("Done. Putting it all together...")
renamer <- function(df, var) {
  df <- df %>%
    rename_with(~ paste0(., ".", var),
                -c("grid", "storm", "storm.rp", "variable"))
  df <- df %>% rename_with(~ var, "variable")
  return(df)
}

storms_wind <- renamer(storms_wind, "u10")
storms_mslp <- renamer(storms_mslp, "mslp")
storms_tp   <- renamer(storms_tp, "tp")

storms <- storms_wind %>%
  inner_join(storms_mslp, by = c("grid", "storm", "storm.rp")) %>%
  inner_join(storms_tp, by = c("grid", "storm", "storm.rp"))

storms$thresh.q <- Q # keep track of threshold used

########### SAVE RESULTS #######################################################
print("Saving...")
write.csv(medians, paste0(WD, "/", "medians.csv"), row.names = FALSE)
write_parquet(metadata, paste0(WD, "/", "storms_metadata.parquet"))
write_parquet(storms, paste0(WD, "/", "storms.parquet"))

cat("\nSaved as:", paste0(WD, "/", "storms.parquet"))
print(paste0("Finished!", length(unique(storms$storm)), " events processed."))

########### END ###############################################################