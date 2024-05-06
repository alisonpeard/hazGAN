# copying /Users/alison/Documents/DPhil/multivariate/_archive/evtGAN_code&data_download04082023/Maxima_rain/evt_rain.R
# Example from vignette
rm(list = ls())
library(arrow)
library(ncdf4)
library(mvPot)
library(SpatialExtremes)

# 100 randomly-selected locations
load('/Users/alison/Documents/DPhil/multivariate/_archive/evtGAN_code&data_download04082023/Maxima/Data/ii.rdata')
coord = read_parquet('/Users/alison/Documents/DPhil/multivariate/era5_data/coords.parquet')
coord = as.matrix(coord[,c('latitude', 'longitude')])



extCoeffBR <- function(h, par){
  s=par[1]
  a=par[2]
  gamma=(h^a)/s
  2 - 2*pnorm(sqrt(gamma)/2)
}
BRIsoFit <- function(data, coord){
  emp <- fitextcoeff(t(data), as.matrix(coord), estim="ST",marge="emp",plot=F)$ext.coeff
  
  BREst <- function(theta,emp) {
    # from Boulaguiem (2020) but modified to match Section 2.1.2?
    s=theta[1]
    alfa=theta[2]
    dist=emp[,1]
    est=emp[,2]
    
    z=sapply(dist, function(x) {
      gamma=(x^alfa)/s
      2*pnorm(sqrt(gamma)/2)
    })
    
    sum((est+z)^2) # changed sign --Alison
  }
  
  list(par=nlminb(start=c(1,1), BREst, lower=c(0,0), upper=c(Inf,2), emp=emp)$par,dist=emp[,1])
} # modified from paper
BRIsoSim <- function(n, par, coord){
  s=par[1]
  alfa=par[2]
  
  vario <- function(h){
    h=as.matrix(h)
    (norm(h,type = "2"))^alfa / s # changed this --Alison
  }
  #loc <- expand.grid(1:4, 1:4)
  simu = simulBrownResnick(n,as.data.frame(coord),vario)
  z=rbind()
  for(i in 1:length(simu)) {
    z=rbind(z,simu[[i]])
  }
  simu=t(z)
}



par <- BRIsoFit(train, coord)$par
sample <- BRIsoSim(1000, par, coord)



######### Boulaguiem (2020) ref ######################################
load('/Users/alison/Documents/DPhil/multivariate/_archive/evtGAN_code&data_download04082023/Maxima/Data/coord_EU.rdata')
coord_EU = as.matrix(coord_EU)
nc_data <- nc_open('/Users/alison/Documents/DPhil/multivariate/_archive/evtGAN_code&data_download04082023/Maxima/Data/temperature_maxima.nc')
lon <- ncvar_get(nc_data, "lon")
lat <- ncvar_get(nc_data, "lat", verbose = F)
t <- ncvar_get(nc_data, "time")
temp <- ncvar_get(nc_data, "MaxTemp")
dim(temp) <- c(18*22,2000)
temp <- temp-273.15
train <- temp[,1:50]
test <- temp[,51:2000]
test_ECs = fitextcoeff(t(test[ii,]), coord_EU[ii,], plot=T)
emp <- fitextcoeff(t(test[ii,]), as.matrix(coord_EU[ii,]), estim = "ST",marge="emp",plot=T)$ext.coeff


BRIsoSim <- function(n,par,coord){
  s=par[1]
  alfa=par[2]
  
  vario <- function(h){
    h=as.matrix(h)
    1/2 * s*(norm(h,type = "2"))^alfa
  }
  #loc <- expand.grid(1:4, 1:4)
  simu = simulBrownResnick(n,as.data.frame(coord),vario)
  z=rbind()
  for(i in 1:length(simu)) {
    z=rbind(z,simu[[i]])
  }
  simu=t(z)
}
extCoeffBR <- function(dist,par){
  lambda=par[1]
  alfa=par[2]
  gamma=lambda*dist^alfa
  2*pnorm(sqrt(gamma)/2)
}

ppp <- BRIsoFit(test[ii,],coord_EU[ii,])
dist_ppp <- ppp$dist
par_temp <- ppp$par
yyy_temp <- sapply(dist_ppp,function(x) extCoeffBR(x,par_temp))

# voriomat for spectral dist
varMat <- function(coord,par){
  vario <- function(h,par){
    lambda=par[1]
    alfa=par[2]
    h=as.matrix(h)
    1/2 * lambda*(norm(h,type = "2"))^alfa
  }
  dim <- nrow(coord)
  coord <- as.data.frame(coord)
  dists <- lapply(1:ncol(coord), function(i) {
    outer(coord[, i], coord[, i], "-")
  })
  computeVarMat <- sapply(1:length(dists[[1]]), function(i) {
    h <- rep(0, ncol(coord))
    for (j in 1:ncol(coord)) {
      h[j] = dists[[j]][i]
    }
    vario(h,par)
  })
  matrix(computeVarMat, dim, dim)
}
varioMat_temp <- varMat(coord_EU,par_temp)