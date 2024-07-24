# Identify events and fit GPD to componentwise maxima. Also assign RP to each event
# INPUT: data_1950_2022.nc
# OUTPUT: (1) event_data.parquet, (2) fitted_data.parquet, (3) monthly_medians.csv
# https://cran.r-project.org/web/packages/eva/eva.pdf
rm(list = ls())
library(eva)
library(arrow)
library(lubridate)
library(dplyr)
require(ggplot2)
library(extRemes)
library(CFtime)
library(tidync)

wd <- "/soge-home/projects/mistral/alison/hazGAN/training"
res <- c(18, 22)
filename <- 'data_1950_2022.nc'
indir <- paste0(wd, paste0('res_', res[1], 'x', res[2]))
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
src <- tidync(paste0(indir, '/', filename))
era5.df <- src %>% hyper_tibble(force = TRUE)
coords <- src %>% activate('grid') %>% hyper_tibble(force = TRUE)
era5.df <- left_join(era5.df, coords, by=c('longitude', 'latitude'))
rm(coords)

era5.df$msl <- -era5.df$msl # negate msl so maximising all vars
era5.df = era5.df[,c('grid', 'time', 'u10', 'msl', 'tp')]

era5.df$time <- as.Date(CFtimestamp(CFtime("days since 1950-01-01", "gregorian", era5.df$time)))
medians <- monthly.medians(era5.df, 'u10')
medians$mslp <- monthly.medians(era5.df, 'msl')$msl
medians$tp <- monthly.medians(era5.df, 'tp')$tp
era5.df$u10 <- standardise.by.month(era5.df, 'u10')
era5.df$msl <- standardise.by.month(era5.df, 'msl')
era5.df$tp <- standardise.by.month(era5.df, 'tp')
########### IDENTIFY STORMS BY WIND SPEEDS ###################################################
df <- aggregate(. ~ time, era5.df[,c('time', 'u10')], r.func) # overall max wind each day

# search for threshold that gives most data
qs <-c(60:99) / 100
rs <- c(1:14)
nclusters <- matrix(nrow=length(rs), ncol=length(qs))
ext.ind <- matrix(nrow=length(rs), ncol=length(qs))
p.vals <- matrix(nrow=length(rs), ncol=length(qs))
for(i in 1:length(rs)){
  for(j in 1:length(qs)){
    thresh <- quantile(df$u10, qs[j])
    d <- decluster(df$u10, thresh=thresh, r=rs[i], method='runs')
    e <- extremalindex(c(d), thresh, r=rs[i], method='runs') # note this is θ=1 a lot, maybe double-check
    p <- Box.test(c(d)[c(d) > thresh], type='Ljung') # independence of exceedences
    nclusters[i, j] <- e[['number.of.clusters']]
    ext.ind[i,j] <- e[['extremal.index']]
    p.vals[i, j] <- p$p.value
  }
}
# remove any cases with dependence in exceedences
nclusters[ext.ind < 0.8] <- 0 # see ?extremalindex, if theta < 1, then there is some dependency (clustering) in the limit
nclusters[p.vals < 0.1] <- 0  #  
ind <- which(nclusters == max(nclusters), arr.ind=TRUE)
r <- rs[ind[1]]
q <- qs[ind[2]]

# make dataframe of events
thresh <- quantile(df$u10, q)

plot(df$u10)
abline(h=thresh, col='red')

declustering <- decluster(df$u10, thresh, r=r)
clusters <- attr(declustering, 'clusters')
times <- df$time[df$u10 > thresh] # times when max(u10) exceeds thresh
u10 <- df$u10[df$u10 > thresh]
cluster.df <- data.frame(time=times, storm=clusters, u10=u10)

# cluster stats
cluster.grouped <- cluster.df %>%
  group_by(storm) %>%
  mutate(storm.size = n()) %>%
  slice(which.max(u10)) %>%
  summarise(u10 = max(u10),
            time = time,
            storm.size = storm.size)

# empirical return period calculation
m <- nrow(cluster.grouped); paste0("Number of storms: ", m)
nyears <- length(unique(year(era5.df$time)))
lambda <- m / nyears; paste0("Yearly rate: ", lambda) # yearly occurrence rate
occurrence.rate <- lambda
p <- 1 - (rank(cluster.grouped$u10, ties.method='average')) / (m + 1) # exceedence probability
rp <- 1 / (lambda*p)
cluster.grouped$storm.rp <- rp
hist(rp, breaks=50)

cluster.df <- left_join(cluster.df, cluster.grouped[c('storm', 'storm.rp', 'storm.size')], by=c('storm'))
cluster.df$q <- q
print('Declustered dataframe:')
head(cluster.df[order(-cluster.df$storm.rp),])
########### FIT GPD TO EVENT MAXIMA ############################################
src <- tidync(paste0(indir, '/', filename))
era5.df.all <- src %>% hyper_tibble()
coords <- src %>% activate('grid') %>% hyper_tibble()
era5.df.all <- left_join(era5.df.all, coords, by=c('longitude', 'latitude'))
era5.df.all$time <- as.Date(CFtimestamp(CFtime("days since 1950-01-01", "gregorian", era5.df.all$time)))
rm(coords)

era5.df.all$msl <- -era5.df.all$msl # negate msl again...
era5.df.all = era5.df.all[,c('grid', 'time', 'u10', 'msl')]
era5.df.all$time <- as.Date(era5.df.all$time)
era5.df.all$u10 <- standardise.by.month(era5.df.all, 'u10')
era5.df.all$msl <- standardise.by.month(era5.df.all, 'msl')
era5.df.all$tp <- standardise.by.month(era5.df.all, 'tp')
grid.cells <- unique(era5.df.all$grid)
ngrid <- length(grid.cells)

# fit GPD models to excesses
q <- 0.8

# WIND
update_progress <- progress_bar(ngrid, "Fitting GPD to wind excesses:", "Complete")
fields <- c("storm", "u10", "time", "storm.rp", "grid", "thresh", "scale", "shape", "p", "ecdf")
wind.transformed.df <- data.frame(matrix(nrow=0, ncol=length(fields)))
colnames(wind.transformed.df) <- fields
for(i in 1:ngrid){
  GRIDCELL <- grid.cells[i]
  
  # process wind
  era5.df <- era5.df.all[era5.df.all$grid == GRIDCELL,]
  era5.df <- era5.df[era5.df$time %in% times,]
  era5.df <- left_join(era5.df, cluster.df[,c('time', 'storm', 'storm.rp')], by=c('time'='time'))
  maxima <- era5.df %>%
    group_by(storm) %>%
    slice(which.max(u10)) %>%
    summarise(u10 = max(u10),
              time = time,
              storm.rp = storm.rp,
              grid = grid)
  thresh.u10 <- quantile(maxima$u10, q)
  
  # valid
  excesses <- maxima$u10[maxima$u10 >= thresh.u10]
  Box.test(excesses) # H0: independent
  hist(excesses)
  
  # fit models...
  new.row <- tryCatch({
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
    maxima
    
  }, error=function(e){
    print(paste0("skipping grid cell ", GRIDCELL, ' ', e))
    missing.columns <- setdiff(names(wind.transformed.df), names(maxima))
    maxima[missing.columns] <- NA
    return(maxima)
  })
  wind.transformed.df <- rbind(wind.transformed.df, new.row)
  update_progress(i)
}

# MSLP
mslp.transformed.df <- data.frame(matrix(nrow=0, ncol=length(fields)))
colnames(mslp.transformed.df) <- fields
update_progress <- progress_bar(ngrid, "Fitting GPD to MSLP excesses:", "Complete")
for(i in 1:ngrid){
  GRIDCELL <- grid.cells[i]
  era5.df <- era5.df.all[era5.df.all$grid == GRIDCELL,]
  era5.df <- era5.df[era5.df$time %in% times,]
  era5.df <- left_join(era5.df, cluster.df[,c('time', 'storm', 'storm.rp')], by=c('time'='time'))
  
  maxima <- era5.df %>%
    group_by(storm) %>%
    slice(which.max(msl)) %>%
    summarise(msl = max(msl),
              time = time,
              storm.rp = storm.rp,
              grid = grid)
  thresh.msl <- quantile(maxima$msl, q)
  
  # validate
  excesses <- maxima$msl[maxima$msl >= thresh.msl]
  Box.test(excesses) # H0: independent
  hist(excesses)
  
  # fit models...
  new.row <- tryCatch({
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
    maxima
    
  }, error=function(e){
    print(paste0("skipping grid cell ", GRIDCELL, e))
    missing.columns <- setdiff(names(mslp.transformed.df), names(maxima))
    maxima[missing.columns] <- NA
    return(maxima)
  })
  mslp.transformed.df <- rbind(mslp.transformed.df, new.row)
  update_progress(i)
} 

# PRECIPITATION
tp.transformed.df <- data.frame(matrix(nrow=0, ncol=length(fields)))
colnames(tp.transformed.df) <- fields
update_progress <- progress_bar(ngrid, "Fitting GPD to precip excesses:", "Complete")
for(i in 1:ngrid){
  GRIDCELL <- grid.cells[i]
  era5.df <- era5.df.all[era5.df.all$grid == GRIDCELL,]
  era5.df <- era5.df[era5.df$time %in% times,]
  era5.df <- left_join(era5.df, cluster.df[,c('time', 'storm', 'storm.rp')], by=c('time'='time'))
  
  maxima <- era5.df %>%
    group_by(storm) %>%
    slice(which.max(tp)) %>%
    summarise(tp = max(tp),
              time = time,
              storm.rp = storm.rp,
              grid = grid)
  thresh.tp <- quantile(maxima$tp, q)
  
  # validate
  excesses <- maxima$tp[maxima$tp >= thresh.tp]
  Box.test(excesses) # H0: independent
  hist(excesses)
  
  # fit models...
  new.row <- tryCatch({
    fit.tp <- gpdAd(maxima$tp[maxima$tp >= thresh.tp],
                     bootstrap = TRUE, bootnum = 10,
                     allowParallel = TRUE, numCores=2) # H0: GPD distribution
    scale <- fit.tp$theta[1]
    shape <- fit.tp$theta[2]
    maxima$thresh <- thresh.tp
    maxima$scale <- scale
    maxima$shape <- shape
    maxima$p <- fit.tp$p.value
    
    # semiparametric transform
    maxima$ecdf <- maxima$tp
    maxima$ecdf[maxima$tp < thresh.tp] <- ecdf(maxima$tp)(maxima$tp[maxima$tp < thresh.tp])
    f.exceed <- 1 - ecdf(maxima$tp)(thresh.tp)
    survival.probs <- 1 - pgpd(maxima$tp[maxima$tp >= thresh.tp], thresh.tp, scale, shape)
    maxima$ecdf[maxima$tp >= thresh.tp] <- 1 - f.exceed * survival.probs
    maxima
    
  }, error=function(e){
    print(paste0("skipping grid cell ", GRIDCELL, e))
    missing.columns <- setdiff(names(tp.transformed.df), names(maxima))
    maxima[missing.columns] <- NA
    return(maxima)
  })
  tp.transformed.df <- rbind(tp.transformed.df, new.row)
  update_progress(i)
} 

# CONCATENATE
transformed.df <- wind.transformed.df %>%
  inner_join(mslp.transformed.df,
             by = c('grid', 'storm'),
             suffix = c('.u10', '.mslp')) %>%
  inner_join(tp.transformed.df,
             by = c('grid', 'storm'),
             suffix = c('', '.tp'))

transformed.df$thresh.q <- q # approx. extremeness measure

########### SAVE TO CSV FOR PYTHON #############################################
write.csv(medians, paste0(indir, '/', 'monthly_medians.csv'), row.names=FALSE)
write_parquet(cluster.df, paste0(indir, '/', 'event_data.parquet'))
write_parquet(transformed.df, paste0(indir, '/', 'fitted_data.parquet'))

print(paste0("Saved as ", indir, '/', 'fitted_data.csv.'))
print(paste0(length(unique(transformed.df$storm)), " events processed."))
########### FIGURES ############################################################
if(FALSE){
  GRIDCELL <- 15
  grid.df <- transformed.df[transformed.df$grid == GRIDCELL,]
  par(mfrow=c(2, 2))
  acf(grid.df$u10, main="U10 cluster maxima ACF")
  pacf(grid.df$u10, main="U10 cluster maxima PACF")
  acf(grid.df$msl, main="MSLP cluster maxima ACF")
  pacf(grid.df$msl, main="MSLP cluster maxima PACF")
}
#####
print(occurrence.rate)
missing.days <- (nyears * 365) - length(unique(era5.df.all$time))
missing.years <- missing.days / 365


