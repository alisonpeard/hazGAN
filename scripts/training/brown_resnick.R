# fit and sample from Brown-Resnick
rm(list = ls())
library(arrow)
library(ncdf4)
library(mvPot)
library(geosphere)
library(SpatialExtremes)

field       <- 1
field.names <- c('u10', 'tp', 'mslp')
ntrain      <- 1000
outdir      <- '/Users/alison/Documents/DPhil/paper1.nosync/results/brown_resnick/'

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

nc_data <- nc_open('/Users/alison/Documents/DPhil/paper1.nosync/training/64x64/data.nc')
lon <- ncvar_get(nc_data, "lon")
lat <- ncvar_get(nc_data, "lat")
coord <- expand.grid(lon = lon, lat = lat)
t <- ncvar_get(nc_data, "time")
U <- ncvar_get(nc_data, "uniform")[field,,,]

# Remove nans
eps          <- 1e-6
U[is.nan(U)] <- eps
U[U == 0]    <- eps
U[U == 1]    <- 1-eps

# figures
heatmap(U[,,1], Rowv=NA, Colv=NA)
nsamples <- dim(U)[3]
dim(U)   <- c(64*64,nsamples)
frechet  <- -1 / log(U) # transform to unit Fréchet
train    <- frechet[,1:ntrain]
test     <- frechet[,(ntrain+1):nsamples]

# load and subset by OP data
op_path <- "/Users/alison/Documents/DPhil/paper1.nosync/training/18x22/ops.parquet"
op_data <- read_parquet(op_path)
dd             <- distm(op_data[,c("lon", "lat")], coord[,c("lon", "lat")])
op_data$idxmin <- apply(dd, 1, which.min)

train <- train[op_data$idxmin,]
test  <- test[op_data$idxmin,]
coord <- coord[op_data$idxmin,]

##### FIT EXTREMAL COEFFICIENTS ################################################
if(TRUE){
  test_ECs  <- fitextcoeff(t(test), as.matrix(coord), estim="ST", marge="frech", plot=T)$ext.coeff
  train_ECs <- fitextcoeff(t(train), as.matrix(coord), estim="ST", marge="frech", plot=T)$ext.coeff
  
  n <- nrow(train)
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
  write_parquet(ECs, sink=paste0(outdir, 'ECs_', field.names[field], '.parquet'))
}
##### SAMPLE FROM FITTED BR ####################################################
ECs <- read_parquet(paste0(outdir, 'ECs_', field.names[field], '.parquet'))
ops <- read_parquet("/Users/alison/Documents/DPhil/paper1.nosync/training/18x22/ops.parquet")
ii <- c(1:nrow(ops))

nsamples <- ntrain
fit <- BRIsoFit(train[ii,], coord[ii,])
par_U <- fit$par

start.time <- Sys.time()
sample <- BRIsoSim(nsamples, par_U, coord[ii,])

end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)

sample_U <- exp(-1/sample) # convert back to uniform distribution
sample.df <- cbind(ops, sample_U)
sample.df[,1:10]
write_parquet(sample.df, paste0(outdir, 'samples_', field.names[field], '.parquet'))

              