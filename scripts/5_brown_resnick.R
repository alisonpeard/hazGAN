# copying /Users/alison/Documents/DPhil/multivariate/_archive/evtGAN_code&data_download04082023/Maxima_rain/evt_rain.R
# Example from vignette
rm(list = ls())
library(arrow)
library(ncdf4)
library(SpatialExtremes)

# 100 randomly-selected locations
load('/Users/alison/Documents/DPhil/multivariate/_archive/evtGAN_code&data_download04082023/Maxima/Data/ii.rdata')

coord = read_parquet('/Users/alison/Documents/DPhil/multivariate/era5_data/coords.parquet')
coord = as.matrix(coord[,c('latitude', 'longitude')])



######### temp data from paper ######################################
nc_data <- nc_open('/Users/alison/Documents/DPhil/multivariate/_archive/evtGAN_code&data_download04082023/Maxima/Data/temperature_maxima.nc')
lon <- ncvar_get(nc_data, "lon")
lat <- ncvar_get(nc_data, "lat", verbose = F)
t <- ncvar_get(nc_data, "time")
temp <- ncvar_get(nc_data, "MaxTemp")
dim(temp) <- c(18*22,2000)
temp <- temp-273.15
train <- temp[,1:50]
test <- temp[,51:2000]

