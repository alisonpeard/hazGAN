"Identify events and fit to marginals.

Includes event metadata such as return period and frequency. Only fit ECDF and 
GPD to 1940-2022 data. Keep 2021 as a holdout test set.

Best run in VS code.

Files:
------
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

4TH LAST RUN: 08-12-2024
3RD:     30-01-2025 Q = 0.9 ==> incoherent results
2ND:     04-02-2024 Q = 0.85
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
library(profvis) # memory profiler

# set up env (depends if running or sourcing script)
try(setwd(getSrcDirectory(function(){})[1]))
try(setwd(dirname(rstudioapi::getActiveDocumentContext()$path)))
source("utils.R")
readRenviron("../../.env")

FILENAME   <- "data_1941_2022.nc"     # nolint
STARTDATE  <- "1941-01-01"
RES        <- c(64, 64)               # nolint
WD         <- Sys.getenv("TRAINDIR")  # nolint
WD         <- paste0(WD, "/", res2str(RES))
RFUNC      <- max                     # nolint, https://doi.org/10.1111/rssb.12498
TEST.YEARS <- c(2021)                 # nolint, exclude from ecdf + gpd fitting
VISUALS    <- TRUE                    # nolint
Q          <- 0.8                     # nolint
DRYRUN     <- FALSE
NDRYRUN    <- 30                      # use nxn gridcells for dry run

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

#deseasonalised <- remove_seasonality(daily, c("u10", "msl", "tp"))
#daily[c("u10", "msl", "tp")] <- deseasonalised$standardised
#medians <- as_tibble(deseasonalised$medians)

std_u10 <- standardise_by_month(daily, "u10")
std_msl <- standardise_by_month(daily, "msl")
std_tp  <- standardise_by_month(daily, "tp")

medians      <- std_u10$medians
medians$mslp <- std_msl$medians$monthly_median
medians$tp   <- std_tp$medians$monthly_median

daily$u10.   <- std_u10$var
daily$msl    <- std_msl$var
daily$tp     <- std_tp$var

end <- Sys.time()
print(end - start)
#%%######## EXTRACT AND TRANSFORM STORMS #######################################
print("Extracting storms...")
start <- Sys.time()
metadata <- storm_extractor(daily, "u10", RFUNC)
end <- Sys.time()
print(end - start)

# fit to marginal data
print("Tranforming fields...")
start <- Sys.time()

# p <- profvis(gpd_transformer(daily, metadata, "u10", Q))
# p
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
end <- Sys.time()
print(end - start)

########### SAVE RESULTS #######################################################
print("Saving...")
if (DRYRUN) stop("Ope nevermind... not saving results for dry run.")
write.csv(medians, paste0(WD, "/", "medians.csv"), row.names = FALSE)
write_parquet(metadata, paste0(WD, "/", "storms_metadata.parquet"))
write_parquet(storms, paste0(WD, "/", "storms.parquet"))

cat("\nSaved as:", paste0(WD, "/", "storms.parquet"))
print(paste0("Finished!", length(unique(storms$storm)), " events processed."))

########### END ###############################################################