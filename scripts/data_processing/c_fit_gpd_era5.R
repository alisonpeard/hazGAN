# https://cran.r-project.org/web/packages/eva/eva.pdf
rm(list = ls())
library(eva)
library(arrow)
library(lubridate)
library(dplyr)
require(ggplot2)
library(extRemes)
filename <- 'data_1971_2022.parquet'
wd <- "/Users/alison/Documents/DPhil/multivariate"
indir <- paste0(wd, '/', 'era5_data')
r.func <- max # https://doi.org/10.1111/rssb.12498
########### DEFINE FUNCTIONS ###################################################
standardise.by.month <- function(df, var){
  df$month <- months(df$time)
  df <- df[,c(var, 'month', 'grid')]
  monthly.median <- aggregate(. ~ month + grid, df, median)
  df$monthly.median <- left_join(df[,c('month', 'grid')], monthly.median, by=c('month'='month', 'grid'='grid'))[[var]]
  df[[var]] <- df[[var]] - df$monthly.median
  return(df[[var]])
}
monthly.medians <- function(df, var){
  df <- df[,c(var, 'time', 'grid')]
  df$month <- months(df$time)
  monthly.median <- aggregate(. ~ month + grid, df[,c(var, 'grid', 'month')], median)
  return(monthly.median)
}
get.ecdf <- function(x){
  return(ecdf(x)(x))
}
progress_bar <- function(n, prefix="", suffix="") {
  pb <- utils::txtProgressBar(min = 0, max = n, style = 3)
  function(i) {
    utils::setTxtProgressBar(pb, i)
    if (i == n) close(pb)
  }
}
########### LOAD DATA AND STANDARDISE IT #######################################
wind.df <- read_parquet(paste0(indir, '/', filename))
wind.df$msl <- -wind.df$msl # negate msl so maximising both vars
wind.df = wind.df[,c('grid', 'time', 'u10', 'msl')]
wind.df$time <- as.Date(wind.df$time)
medians <- monthly.medians(wind.df, 'u10')
medians$mslp <- monthly.medians(wind.df, 'msl')$msl
wind.df$u10 <- standardise.by.month(wind.df, 'u10')
wind.df$msl <- standardise.by.month(wind.df, 'msl')
########### DEFINE EVENTS ######################################################
wind.df$u10 <- scale(wind.df$u10)
wind.df$msl <- scale(wind.df$msl)
df <- aggregate(. ~ time, wind.df[,c('time', 'u10', 'msl')], r.func)
colnames(df) <- c('time', 'u10', 'msl')
df$ecdf <- get.ecdf(apply(df[,c('u10', 'msl')], 1, r.func)); hist(df$ecdf)

# search for threshold that gives most data
qs <- c(60:99) / 100
rs <- c(1:14)
nclusters <- matrix(nrow=length(rs), ncol=length(qs))
ext.ind <- matrix(nrow=length(rs), ncol=length(qs))
for(i in 1:length(rs)){
  for(j in 1:length(qs)){
    thresh <- quantile(df$ecdf, qs[j])
    d <- decluster(df$ecdf, thresh=thresh, r=rs[i], method='runs')
    e <- extremalindex(c(d), thresh, r=rs[i], method='runs')
    nclusters[i, j] <- e[['number.of.clusters']]
    ext.ind[i,j] <- e[['extremal.index']]
  }
}
nclusters[ext.ind < 0.8] <- 0
ind <- which(nclusters == max(nclusters), arr.ind=TRUE)
r <- rs[ind[1]]
q <- qs[ind[2]]

thresh <- quantile(df$ecdf, q)
declustering <- decluster(df$ecdf, thresh, r=r)
times <- df$time[df$ecdf > thresh] # times when joint extremeness exceeds thresh
ecdfs <- df$ecdf[df$ecdf > thresh]
clusters <- attr(declustering, 'clusters')
cluster.df <- data.frame(time=times, cluster=clusters, ecdf=ecdfs)
cluster.df <- cluster.df %>%
  group_by(cluster) %>%
  mutate(cluster.size = n()) %>%
  mutate(ecdf = mean(ecdf)) %>%
  ungroup()
cluster.df$q <- q
########### FIT GPD TO EVENT MAXIMA ############################################
wind.df.all <- read_parquet(paste0(indir, '/', filename))
wind.df.all$msl <- -wind.df.all$msl # negate msl again...
wind.df.all = wind.df.all[,c('grid', 'time', 'u10', 'msl')]
wind.df.all$time <- as.Date(wind.df.all$time)
wind.df.all$u10 <- standardise.by.month(wind.df.all, 'u10')
wind.df.all$msl <- standardise.by.month(wind.df.all, 'msl')
grid.cells <- unique(wind.df.all$grid)
ngrid <- length(grid.cells)

# NOTE: run again to check whether these are needed and delete if not
# p.vals <- vector(length=ngrid)
# threshs <- vector(length=ngrid)
# shape.params <- vector(length=ngrid)
# scale.params <- vector(length=ngrid)

# fit GPD models to excesses
q <- 0.8
update_progress <- progress_bar(ngrid, "Fitting GPD to wind excesses:", "Complete")
wind.transformed.df <- data.frame()
for(i in 1:ngrid){
  GRIDCELL <- grid.cells[i]
  # process wind
  wind.df <- wind.df.all[wind.df.all$grid == GRIDCELL,]
  wind.df <- wind.df[wind.df$time %in% times,]
  
  wind.df <- left_join(wind.df, cluster.df[,c('time', 'cluster', 'ecdf')], by=c('time'='time'))
  colnames(wind.df)[colnames(wind.df) == 'ecdf'] <- 'extremeness'
  #wind.df$cluster <- cluster.df$cluster
  #wind.df$extremeness <- cluster.df$ecdf

  maxima <- wind.df %>%
    group_by(cluster) %>%
    slice(which.max(u10)) %>%
    summarise(u10 = max(u10),
              time = time,
              extremeness = extremeness,
              grid = grid)
  thresh.u10 <- quantile(maxima$u10, q)
  
  # valid
  excesses <- maxima$u10[maxima$u10 >= thresh.u10]
  Box.test(excesses) # H0: independent
  hist(excesses)
  
  # fit models...
  tryCatch({
    fit.u10 <- gpdAd(maxima$u10[maxima$u10 >= thresh.u10],
                     bootstrap = TRUE, bootnum = 10,
                     allowParallel = TRUE, numCores=2) # H0: GPD distribution
    scale <- fit.u10$theta[1]
    shape <- fit.u10$theta[2]
    maxima$thresh <- thresh.u10
    maxima$scale <- scale
    maxima$shape <- shape
    maxima$p <- fit.u10$p.value
    
    # semiparametric transform
    maxima$ecdf <- maxima$u10
    maxima$ecdf[maxima$u10 < thresh.u10] <- ecdf(maxima$u10)(maxima$u10[maxima$u10 < thresh.u10])
    f.exceed <- 1 - ecdf(maxima$u10)(thresh.u10)
    survival.probs <- f.exceed * (1 - pgpd(maxima$u10[maxima$u10 >= thresh.u10], thresh.u10, scale, shape))
    maxima$ecdf[maxima$u10 >= thresh.u10] <- 1 - survival.probs
    wind.transformed.df <- rbind(wind.transformed.df, maxima)
    
  }, error=function(e){
    print(paste0("skipping grid cell ", GRIDCELL, e))
  })
  update_progress(i)
}

mslp.transformed.df <- data.frame()
update_progress <- progress_bar(ngrid, "Fitting GPD to MSLP excesses:", "Complete")
for(i in 1:ngrid){
  GRIDCELL <- grid.cells[i]
  wind.df <- wind.df.all[wind.df.all$grid == GRIDCELL,]
  wind.df <- wind.df[wind.df$time %in% times,]
  wind.df <- left_join(wind.df, cluster.df[,c('time', 'cluster', 'ecdf')], by=c('time'='time'))
  colnames(wind.df)[colnames(wind.df) == 'ecdf'] <- 'extremeness'
  #wind.df$cluster <- cluster.df$cluster
  #wind.df$extremeness <- cluster.df$ecdf
  
  maxima <- wind.df %>%
    group_by(cluster) %>%
    slice(which.max(msl)) %>%
    summarise(msl = max(msl),
              time = time,
              extremeness = extremeness,
              grid = grid)
  thresh.msl <- quantile(maxima$msl, q)
  
  # validate
  excesses <- maxima$msl[maxima$msl >= thresh.msl]
  Box.test(excesses) # H0: independent
  hist(excesses)
  
  # fit models...
  tryCatch({
    fit.msl <- gpdAd(maxima$msl[maxima$msl >= thresh.msl],
                     bootstrap = TRUE, bootnum = 10,
                     allowParallel = TRUE, numCores=2) # H0: GPD distribution
    scale <- fit.msl$theta[1]
    shape <- fit.msl$theta[2]
    maxima$thresh <- thresh.msl
    maxima$scale <- scale
    maxima$shape <- shape
    maxima$p <- fit.msl$p.value
    
    # semiparametric transform
    maxima$ecdf <- maxima$msl
    maxima$ecdf[maxima$msl < thresh.msl] <- ecdf(maxima$msl)(maxima$msl[maxima$msl < thresh.msl])
    f.exceed <- 1 - ecdf(maxima$msl)(thresh.msl)
    survival.probs <- 1 - pgpd(maxima$msl[maxima$msl >= thresh.msl], thresh.msl, scale, shape)
    maxima$ecdf[maxima$msl >= thresh.msl] <- 1 - f.exceed * survival.probs
    mslp.transformed.df <- rbind(mslp.transformed.df, maxima)
    
  }, error=function(e){
    print(paste0("skipping grid cell ", GRIDCELL, e))
  })
  update_progress(i)
} 

transformed.df <- wind.transformed.df %>% inner_join(mslp.transformed.df,
                                                     by = c('grid', 'cluster'),
                                                     suffix = c('.u10', '.mslp'))
transformed.df$thresh.q <- q # approx. extremeness measure
########### SAVE TO CSV FOR PYTHON #############################################
write.csv(medians, paste0(indir, '/', 'monthly_medians.csv'), row.names=FALSE)
write_parquet(cluster.df, paste0(indir, '/', 'event_data.parquet'))
write_parquet(transformed.df, paste0(indir, '/', 'fitted_data.parquet'))

print(paste0("Saved as ", indir, '/', 'fitted_data.csv.'))
print(paste0(length(unique(transformed.df$cluster)), " events processed."))
########### FIGURES ############################################################
GRIDCELL <- 15
grid.df <- transformed.df[transformed.df$grid == GRIDCELL,]
par(mfrow=c(2, 2))
acf(grid.df$u10, main="U10 cluster maxima ACF")
pacf(grid.df$u10, main="U10 cluster maxima PACF")
acf(grid.df$msl, main="MSLP cluster maxima ACF")
pacf(grid.df$msl, main="MSLP cluster maxima PACF")
