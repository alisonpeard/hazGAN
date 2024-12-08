"Identify events and fit GPD to marginal exceedances.

Add event metadata such as return period and frequency.

Files:
-----
- input: data_1940_2022.nc
- output:
  - event_data.parquet
  - fitted_data.parquet
  - monthly_medians.csv

TODO:
-----
- train-test split
- convert to Python!

LAST RUN: 06-12-2024
"
rm(list = ls())
library(eva)
library(arrow)
library(lubridate)
library(dplyr)
require(ggplot2)
library(extRemes)
library(CFtime)
library(tidync)

source("utils.R")
readRenviron("../../.env")

FILENAME <- 'data_1940_2022.nc'
WD <- Sys.getenv("TRAINDIR")
RFUNC <- max # https://doi.org/10.1111/rssb.12498
TEST.YEAR <- 2022 # exclude from GPD fit
VISUALS <- TRUE
Q <- 0.8

########### LOAD AND STANDARDISE DATA ##########################################
src <- tidync(paste0(WD, '/', FILENAME))
daily <- src %>% hyper_tibble(force = TRUE)
coords <- src %>% activate('grid') %>% hyper_tibble(force = TRUE)
daily <- left_join(daily, coords, by=c('lon', 'lat'))
rm(coords)

daily$msl <- -daily$msl # negate pressure so maximizing all vars
daily = daily[,c('grid', 'time', 'u10', 'msl', 'tp')]

daily$time <- as.Date(CFtimestamp(
  CFtime("days since 1940-01-01", "gregorian", daily$time)
  ))

medians <- monthly.medians(daily, 'u10')
medians$mslp <- monthly.medians(daily, 'msl')$msl
medians$tp <- monthly.medians(daily, 'tp')$tp

daily$u10 <- standardise.by.month(daily, 'u10')
daily$msl <- standardise.by.month(daily, 'msl')
daily$tp <- standardise.by.month(daily, 'tp')

########### EXTRACT AND TRANSFORM STORMS #######################################
# identify storms by max wind speeds
metadata <- storm.extractor(daily, 'u10', RFUNC)

# fit to marginal data
daily <- daily[daily$time %in% times,]
storms.wind <- gpd.transformer(daily, 'u10', Q)
storms.mslp <- gpd.transformer(daily, 'msl', Q)
storms.tp   <- gpd.transformer(daily, 'tp', Q)

renamer <- function(df, var){
  df <- df %>%
    rename_with(~ paste0(., '.', var), -c(grid, storm, storm.rp, variable))
  df <- df %>% rename_with(~ var, variable)
  return(df)
}
renamer(storms.wind, 'u10')

storms.wind <- renamer(storms.wind, 'u10')
storms.mslp <- renamer(storms.mslp, 'mslp')
storms.tp   <- renamer(storms.tp, 'tp')

storms <- storms.wind %>%
  inner_join(storms.mslp, by = c('grid', 'storm', 'storm.rp')) %>%
  inner_join(storms.tp, by = c('grid', 'storm', 'storm.rp'))

storms$thresh.q <- Q # approx. extremeness measure

########### SAVE RESULTS #######################################################
write.csv(medians, paste0(WD, '/', 'medians.csv'), row.names=FALSE)
write_parquet(metadata, paste0(WD, '/', 'storms_metadata.parquet'))
write_parquet(storms, paste0(WD, '/', 'storms.parquet'))

cat("\nSaved as:", paste0(WD, '/', 'storms.parquet'))
print(paste0(length(unique(storms$storm)), " events processed."))

########### FIGURES ############################################################
if(VISUALS){
  GRIDCELL <- 15
  gridcell <- storms[storms$grid == GRIDCELL,]
  par(mfrow=c(2, 2))
  acf(gridcell$u10, main="U10 cluster maxima ACF")
  pacf(gridcell$u10, main="U10 cluster maxima PACF")
  acf(gridcell$msl, main="MSLP cluster maxima ACF")
  pacf(gridcell$msl, main="MSLP cluster maxima PACF")
}

print(occurrence.rate)
missing.days <- (nyears * 365) - length(unique(daily$time))
missing.years <- missing.days / 365

########### END ################################################################