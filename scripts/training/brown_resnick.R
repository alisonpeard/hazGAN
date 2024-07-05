# fit and sample from Brown-Resnick
rm(list = ls())
library(arrow)
library(ncdf4)
library(mvPot)
library(SpatialExtremes)

channel <- 1
channel.names <- c('u10', 'mslp')
ntrain <- 1000
outdir <- '/Users/alison/Documents/DPhil/multivariate/results/brown_resnick/'

extCoeffBR <- function(h, par){
  s=par[1]
  a=par[2]
  gamma=(h^a)*s
  2*pnorm(sqrt(gamma)/2)
} # modified from https://doi.org/10.48550/arXiv.2111.00267
BRIsoFit <- function(data,coord){
  emp <- fitextcoeff(t(data), as.matrix(coord), estim="ST", marge="frech",plot=F)$ext.coeff
  
  BREst <- function(theta, emp) {
    s=theta[1]
    alfa=theta[2]
    dist=emp[,1]
    est=emp[,2]
    
    z=sapply(dist, function(h) {
      gamma=(h^alfa)*s
      2*pnorm(sqrt(gamma)/2) # changed
    })
    
    sum((est-z)^2)
  }
  
  list(par=nlminb(start=c(1,1), BREst, lower=c(0,0), upper=c(Inf,2), emp=emp)$par, dist=emp[,1])
}
BRIsoSim <- function(n, par, coord){
  s=par[1]
  alfa=par[2]
  
  vario <- function(h){
    h=as.matrix(h)
    (norm(h, type="2"))^alfa * s # changed this --Alison
  }
  simu = simulBrownResnick(n,as.data.frame(coord),vario)
  z=rbind()
  for(i in 1:length(simu)) {
    z=rbind(z,simu[[i]])
  }
  simu=t(z)
}

coord <- read_parquet('/Users/alison/Documents/DPhil/multivariate/era5_data/coords.parquet')
coord <- as.matrix(coord[,c('latitude', 'longitude')])
nc_data <- nc_open('/Users/alison/Documents/DPhil/multivariate/era5_data/data.nc')
lon <- ncvar_get(nc_data, "lon")
lat <- ncvar_get(nc_data, "lat")
t <- ncvar_get(nc_data, "time")
U <- ncvar_get(nc_data, "uniform")[channel,,,]
heatmap(U[,,1], Rowv=NA, Colv=NA)

dim(U) <- c(18*22,2715)
frechet <- -1 / log(U) # transform to unit Fréchet
train <- frechet[,1:ntrain]
test <- frechet[,(ntrain+1):2715]
##### FIT EXTREMAL COEFFICIENTS ################################################
if(TRUE){
  test_ECs <- fitextcoeff(t(test), as.matrix(coord), estim="ST", marge="frech", plot=T)$ext.coeff
  train_ECs <- fitextcoeff(t(train), as.matrix(coord), estim="ST", marge="frech", plot=T)$ext.coeff
  
  n <- 396
  npairs <- n * (n - 1) / 2 # i.e. (n C 2)
  indices <- list()
  for(i in 1:(n-1)){
    for(j in (i+1):n){
      indices <- rbind(indices, c(i, j))
    }
  }
  
  # save for later
  ECs <- cbind(indices, test_ECs)
  ECs <- cbind(ECs, train_ECs[,'ext.coeff'])
  colnames(ECs) <- c('i', 'j', 'distance', 'test_EC', 'train_EC')
  ECs <- data.frame(t(apply(ECs, 1, unlist)))
  write_parquet(ECs, sink=paste0(outdir, 'ECs_', channel.names[channel], '.parquet'))
}
##### SAMPLE FROM FITTED BR ####################################################
ECs <- read_parquet(paste0(outdir, 'ECs_', channel.names[channel], '.parquet'))
ops <- read_parquet("/Users/alison/Documents/DPhil/multivariate/era5_data/ops.parquet")
ii <- ops$grid

nsamples <- ntrain
fit <- BRIsoFit(train[ii,], coord[ii,])
par_U <- fit$par
sample <- BRIsoSim(nsamples, par_U, coord[ii,])
sample_U <- exp(-1/sample) # convert back to uniform distribution
sample.df <- cbind(ops, sample_U)
sample.df[,1:10]
write_parquet(sample.df, paste0(outdir, 'samples_', channel.names[channel], '.parquet'))

              