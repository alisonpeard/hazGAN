library(arrow)
library(eva)
library(zoo)

FIELD  <- 'u10'
storms <- read_parquet("/Users/alison/Documents/DPhil/paper1.nosync/training/64x64/storms.parquet")
idx    <- as.integer(runif(20, 1, 4096))
thresholds <- seq(0.75, 0.99, length.out=20)
bootnums <- c(1:20)

# single loop pass
i <- idx[1]
gridcell <- storms[storms$grid == i,]
var       <- gridcell[[FIELD]]

shapes <- matrix(nrow=length(thresholds), ncol=length(bootnums))
pvals  <- matrix(nrow=length(thresholds), ncol=length(bootnums))

set.seed(42)
for (t in 1:length(thresholds)) {
  thresh    <- thresholds[t]
  loc       <- quantile(var, thresh)
  tail      <- var[var > loc]
  # hist(tail)
  for(b in 1:length(bootnums)) {
    fit       <- gpdAd(tail, bootstrap=TRUE, bootnum=bootnums[b], allowParallel=TRUE, numCores=3)
    print(paste0("eva::gpdAd fit for threshold ", thresh, ": ", fit))
    shapes[t,b] <- fit$theta[2]
    pvals[t,b]  <- fit$p.value
  }
}


locs <- quantile(var, thresholds)
fits <- gpdSeqTests(var, thresholds=locs, method="ad", nsim=10, allowParallel=TRUE, numCores=6)

mrlplot(var, main="Mean Residual Life Plot")
plot(fits)


par(mfrow=c(2,2))
plot(fits$threshold, fits$est.shape, type="b", 
     xlab="Threshold", ylab="Shape Parameter",
     main="Shape Parameter Stability")
abline(h=0, lty=2, col="red")


plot(fits$threshold, fits$est.scale, type="b",
     xlab="Threshold", ylab="Modified Scale",
     main="Modified Scale Parameter Stability")


plot(fits$threshold, fits$p.values, type="b",
     xlab="Threshold", ylab="P-value",
     main="P-values vs Threshold")
abline(h=0.05, lty=2, col="red")


plot(fits$threshold, fits$num.above, type="b",
     xlab="Threshold", ylab="Number of Exceedances",
     main="Sample Size vs Threshold")
par(mfrow=c(1,1))


tcplot(var, tlim=c(4,11))

################################################################################
select_gpd_threshold <- function(var, min_exceedances=30, nthresholds=50, nsim=20) {
  thresholds <- quantile(var, probs=seq(0.7, 0.999, length.out=nthresholds))
  
  fits <- gpdSeqTests(var, thresholds=thresholds, method="ad", nsim=nsim)
  
  valid_n          <- fits$num.above >= min_exceedances
  valid_stops      <- fits$ForwardStop <1 & fits$StrongStop < 1
  shape_stability  <- rollapply(fits$est.shape, width=3, FUN=sd, fill=NA)
  valid_stability  <- !is.na(shape_stability < Inf)
  valid_thresholds <- which(valid_n & valid_stops & valid_stability)
  
  if(length(valid_thresholds) > 0) {
    #idx <- which.min(fits$est.shape)
    idx <- min(valid_thresholds)
    return(list(
      loc   = fits$threshold[idx],
      theta = c(fits$est.scale[idx], fits$est.shape[idx]),
      p.value = fits$p.values[idx],
      n_exceed = fits$num.above[idx]
    ))
  } else {
    return(NULL)
  }
}

result <- select_threshold(var)
