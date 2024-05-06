# fit and sample from Brown-Resnick
rm(list = ls())
library(arrow)
library(ncdf4)
library(mvPot)
library(SpatialExtremes)

channel <- 1
ntrain <- 1000
outdir <- '/Users/alison/Documents/DPhil/multivariate/results/brown_resnick/'

extCoeffBR <- function(h, par){
  s=par[1]
  a=par[2]
  gamma=(h^a)/s
  2 - 2*pnorm(sqrt(gamma)/2)
} # from https://doi.org/10.48550/arXiv.2111.00267
BRIsoFit <- function(data,coord){
  emp <- fitextcoeff(t(data), as.matrix(coord), estim="ST", marge="emp",plot=F)$ext.coeff
  
  BREst <- function(theta,emp) {
    lambda=theta[1]
    alfa=theta[2]
    dist=emp[,1]
    est=emp[,2]
    
    z=sapply(dist, function(x) {
      gamma=(x^alfa)*lambda
      2*pnorm(sqrt(gamma)/2)
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

load(paste0(outdir, 'ii.rdata'))
coord = read_parquet('/Users/alison/Documents/DPhil/multivariate/era5_data/coords.parquet')
coord = as.matrix(coord[,c('latitude', 'longitude')])
nc_data <- nc_open('/Users/alison/Documents/DPhil/multivariate/era5_data/data.nc')
lon <- ncvar_get(nc_data, "lon")
lat <- ncvar_get(nc_data, "lat")
t <- ncvar_get(nc_data, "time")
U <- ncvar_get(nc_data, "U")[channel,,,]
dim(U) <- c(18*22,2715)
frechet <- -1 / log(U) # transform to unit Fréchet
train <- frechet[,1:ntrain]
test <- frechet[,(ntrain+1):2715]

# fit ECs
emp <- fitextcoeff(t(test[ii,]), as.matrix(coord[ii,]), estim="ST", marge="frech", plot=T)#$ext.coeff
ppp <- BRIsoFit(test[ii,], coord[ii,])
dist_ppp <- ppp$dist
par_U <- ppp$par;par_U
yyy_U <- sapply(dist_ppp, function(x) extCoeffBR(x, par_U))
plot(dist_ppp, yyy_U)

# sample new to compare
nsamples <- 2
par <- BRIsoFit(train, coord)$par # fit on train sample then sample new
sample <- BRIsoSim(nsamples, par_U, coord)
dim(sample) <- c(22, 18, nsamples)
sample_U <- exp(-1/sample)
lat <- unique(coord[,1])
lon <- unique(coord[,2])
lat <- ncdim_def("latitude", "degrees_north", lat)
lon <- ncdim_def("longitude", "degrees_east", lon)
index <- ncdim_def('sample', 'index', 1:nsamples, unlim=TRUE)
mv <- -999
var <- ncvar_def("marginals", "uniform", list(lon, lat, index), longname="Brown-Resnick samples of U10 with uniform marginals", mv)

channels = c('u10', 'mslp')
filename <- paste0(outdir, 'sample_', channels[channel], '.nc')

ncnew <- nc_create(filename, list(var))
ncvar_put(ncnew, var, sample_U)
nc_close(ncnew)

ncnew <- nc_open(filename)
dim(ncvar_get(ncnew, "marginals"))
