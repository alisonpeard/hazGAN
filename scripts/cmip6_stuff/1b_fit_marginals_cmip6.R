# https://cran.r-project.org/web/packages/eva/eva.pdf
# look into latent space models to enforce smooth variation across spatial domain
library(eva)
library(lubridate)
library(dplyr)
require(ggplot2)
library(extRemes)

standardise.by.month <- function(df, var){
  df <- df[,c(var, 'time')]
  df$month <- months(df$time)
  monthly.median <- aggregate(. ~ month, df, median)
  df$monthly.median <- monthly.median[match(df$month, monthly.median$month), var]
  df[[var]] <- df[[var]] - df$monthly.median
  return(df[[var]])
}
get.ecdf <- function(x){
  return(ecdf(x)(x))
}

wd <- "/Users/alison/Documents/DPhil/multivariate"
indir <- paste0(wd, '/', 'cmip6_data')

# load data and impute missing data
if(TRUE){
  # read in the wind data
  wind.df <- read.csv(paste0(indir, '/', 'u10_1850_2014.csv'))
  wind.df <- aggregate(. ~ time, wind.df, mean)
  wind.df = wind.df[,c('time', 'u10')]
  wind.df$time <- as.Date(wind.df$time)
  wind.df$u10 <- standardise.by.month(wind.df, 'u10')
  
  # read in the pressure data
  slp.df <- read.csv(paste0(indir, '/', 'mslp_1850_2014.csv'))
  slp.df <- aggregate(. ~ time, slp.df, mean)
  slp.df = slp.df[,c('time', 'mslp')]
  slp.df$time <- as.Date(slp.df$time)
  slp.df$mslp <- -standardise.by.month(slp.df, 'mslp') # NB: note the minus sign
}

# define events using aggregate statistic
if(TRUE){
  # join into one df based-on times (discard missing times)
  df <- wind.df
  df <- merge(df, slp.df, by='time')

  # aggreate risk score still undecided
  #df$ecdf <- 0.5 * (get.ecdf(df$u10) + get.ecdf(df$mslp)); # converges to normal => not ideal
  #df$scaled <- 0.5 * (scale(df$u10) + scale(df$mslp))
  df$ecdf <- get.ecdf(scale(df$u10) + scale(df$mslp)); hist(df$ecdf)
  
  q <- 0.8
  thresh <- quantile(df$ecdf, q)
  declustering <- decluster(df$ecdf, thresh, method='interval');declustering
  times <- df$time[df$ecdf > thresh]
  ecdfs <- df$ecdf[df$ecdf > thresh]
  clusters <- attr(declustering, 'clusters')
  cluster.df <- data.frame(time=times, cluster=clusters, ecdf=ecdfs)
  cluster.df <- cluster.df %>%
    group_by(cluster) %>%
    mutate(cluster.size = n()) %>%
    mutate(ecdf = mean(ecdf)) %>%
    ungroup()
  
  # evalutation and plots
  if(FALSE){
    df$excess <- c(declustering)
    df$exceedance <- as.factor(df$excess >= thresh)
    df$year <- year(df$time)
    YEARS <- seq(1990, 2000)
    qplot(time, excess, data=df[is.element(df$year, YEARS),], colour=exceedance)

    excesses <- df$excess[df$excess >= thresh]
    print(declustering)
    print(length(excesses))
    par(mfrow=c(1, 2))
    hist(excesses);plot(excesses, pch=20)
    
    times <- df$time[df$excess >= thresh]
    u10s <- df$u10[df$excess >= thresh]
    mslps <- df$mslp[df$excess >= thresh]
    ones <- rep(1, length(clusters))
    events.df <- data.frame(time=times, u10=u10s, mslp=mslps, z=excesses, count=ones, cluster=clusters)
    
    # plot partial duration series
    par(mfrow=c(2, 1))
    plot(events.df$time, events.df$u10, pch=20)
    plot(events.df$time, events.df$mslp, pch=20)
    
    # histograms
    par(mfrow=c(1, 2))
    hist(events.df$u10);hist(events.df$mslp)
    
    # compare extremal relationships
    par(mfrow=c(3,1))
    plot(events.df$z, events.df$u10, pch=20)
    plot(events.df$z, events.df$mslp, pch=20)
    plot(events.df$u10, events.df$mslp, pch=20)
    
    # look at stationarity
    Box.test(events.df$u10) # fails
    Box.test(events.df$mslp) # fails
    par(mfrow=c(2,1))
    plot(times, u10s, pch=20)
    plot(times, mslps, pch=20)
    par(mfrow=c(2,2))
    acf(u10s);pacf(u10s)
    acf(mslps);pacf(mslps)

    # now get cluster statistics
    cluster.counts <- aggregate(count ~ cluster, events.df, sum) # mean cluster size
    min(cluster.counts$count);mean(cluster.counts$count);max(cluster.counts$count)
    ncluster <- max(cluster.counts$cluster)
    
    # get componentwise maxima over clusters and test independence
    maxima <- aggregate(u10 ~ cluster, events.df, max)
    maxima$precip <- aggregate(precip ~ cluster, events.df, max)$precip
    Box.test(maxima$u10)
    Box.test(maxima$precip)
    par(mfrow=c(2, 1))
    plot(maxima$u10, type='l');plot(maxima$precip, type='l')
    par(mfrow=c(1,1))
    plot(maxima$u10, maxima$precip, pch=20)
    # GEV fits very well unsure of asymptotic justification for variable block size
    # i.e. different distribution so not iid
    # u10.fit <- gevrFit(maxima$u10);u10.params <- u10.fit$par.ests;plot(u10.fit)
    
    # Try GPD on aggregated wind and precip components
    if(FALSE){
      quantiles <- seq(.7, .999, .005)
      thresholds <- quantile(maxima$u10, quantiles)
      res <- gpdSeqTests(maxima$u10, thresholds = thresholds, method ="ad", nsim = 10, allowParallel=TRUE)
      
      # visual choice of threshold
      plot(thresholds, res$p.values, type='l')
      abline(h=0.1, col='red')
      mrlplot(maxima$u10)
      plot(thresholds, res$est.scale, type='l')
      plot(thresholds, res$est.shape, type='l')
      
      q = 0.75
      thresh <- quantile(maxima$u10, q)
      fit.u10 <- gpdFit(maxima$u10, threshold = thresh)
      plot(fit.u10)
      fit.u10$par.ests
    }
  }
    
  rm(df, wind.df, slp.df, thresh, declustering)
}

# fit GPD over componentwise maxima per grid cell
wind.df.all <- read.csv(paste0(indir, '/', 'u10_1850_2014.csv'))
wind.df.all <- aggregate(. ~ grid + time, wind.df.all, max)
slp.df.all <- slp.df <- read.csv(paste0(indir, '/', 'mslp_1850_2014.csv'))
slp.df.all <- aggregate(. ~ grid + time, slp.df.all, max)
grid.cells <- unique(wind.df.all$grid)

ngrid <- length(grid.cells)
p.vals <- vector(length=ngrid)
threshs <- vector(length=ngrid)
shape.params <- vector(length=ngrid)
scale.params <- vector(length=ngrid)

# process variables
q <- 0.9
wind.transformed.df <- data.frame()
slp.transformed.df <- data.frame()
for(i in 1:ngrid){
  GRIDCELL <- grid.cells[i]
  # process wind
  if(TRUE){
    wind.df <- wind.df.all[wind.df.all$grid == GRIDCELL,]
    wind.df = wind.df[,c('grid', 'time', 'u10')]
    wind.df$time <- as.Date(wind.df$time)
    wind.df$u10 <- standardise.by.month(wind.df, 'u10')
    wind.df <- wind.df[wind.df$time %in% times,]
  }
  
  wind.df$cluster <- cluster.df$cluster
  wind.df$extremeness <- cluster.df$ecdf

  maxima <- wind.df %>%
    group_by(cluster) %>%
    summarise(u10 = max(u10),
              time = time[which.max(u10)],
              extremeness = extremeness[which.max(u10)],
              grid = grid[which.max(u10)])
  
  # choose a threshold and check exceedences not clustered
  thresh.u10 <- quantile(maxima$u10, q)
  
  # summary stats and plots
  if(FALSE){
    cluster.counts <- data.frame(size=rep(1, length(clusters)), cluster=clusters)
    cluster.counts <- aggregate(size ~ cluster, cluster.counts, sum)
    maxima <- merge(maxima.wind, maxima.slp, by="cluster")
    maxima$lag <- as.numeric(maxima$time.slp - maxima$time.u10)
    
    # plot cluster sizes and lags
    par(mfrow=c(2, 1))
    hist(cluster.counts$size)
    hist(maxima$lag)
    
    par(mfrow=c(1, 2))
    hist(maxima$u10, breaks=30)
    hist(maxima$slp, breaks=30)
    
    par(mfrow=c(2, 2))
    acf(maxima$u10);pacf(maxima$u10);
    acf(maxima$slp);pacf(maxima$slp);
    
    # plot scatterplot and check exceedences
    par(mfrow=c(1, 1))
    plot(maxima$u10, maxima$slp, pch=20)
    abline(v=thresh.u10, col='red', lwd=2, lty='dashed')
    abline(h=thresh.slp, col='red', lwd=2, lty='dashed')
    
    # check exceedences independent
    Box.test(maxima$u10[maxima$u10 >= thresh.u10])
    Box.test(maxima$slp[maxima$slp >= thresh.slp])
  }
  
  # fit models...
  fit.u10 <- gpdAd(maxima$u10[maxima$u10 >= thresh.u10])
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
  maxima$ecdf[maxima$u10 >= thresh.u10] <- 1 - f.exceed * pgpd(maxima$u10[maxima$u10 >= thresh.u10], thresh.u10, scale, shape)
  wind.transformed.df <- rbind(wind.transformed.df, maxima)
}
for(i in 1:ngrid){
  GRIDCELL <- grid.cells[i]
  # process sea-level pressure
  if(TRUE){
    slp.df <- slp.df.all[slp.df.all$grid == GRIDCELL,]
    slp.df = slp.df[,c('grid', 'time', 'mslp')]
    slp.df$time <- as.Date(slp.df$time)
    slp.df$mslp <- -standardise.by.month(slp.df, 'mslp')
    slp.df <- slp.df[slp.df$time %in% times,]
  }
  
  slp.df$cluster <- cluster.df$cluster
  slp.df$extremeness <- cluster.df$ecdf
  
  maxima <- slp.df %>%
    group_by(cluster) %>%
    summarise(mslp = max(mslp),
              time = time[which.max(mslp)],
              extremeness = extremeness[which.max(mslp)],
              grid = grid[which.max(mslp)])
  
  # choose a threshold and check exceedences not clustered
  thresh.mslp <- quantile(maxima$mslp, q)
  
  # summary stats and plots
  if(FALSE){
    cluster.counts <- data.frame(size=rep(1, length(clusters)), cluster=clusters)
    cluster.counts <- aggregate(size ~ cluster, cluster.counts, sum)
    maxima <- merge(maxima.wind, maxima.slp, by="cluster")
    maxima$lag <- as.numeric(maxima$time.slp - maxima$time.mslp)
    
    # plot cluster sizes and lags
    par(mfrow=c(2, 1))
    hist(cluster.counts$size)
    hist(maxima$lag)
    
    par(mfrow=c(1, 2))
    hist(maxima$mslp, breaks=30)
    hist(maxima$slp, breaks=30)
    
    par(mfrow=c(2, 2))
    acf(maxima$mslp);pacf(maxima$mslp);
    acf(maxima$slp);pacf(maxima$slp);
    
    # plot scatterplot and check exceedences
    par(mfrow=c(1, 1))
    plot(maxima$mslp, maxima$slp, pch=20)
    abline(v=thresh.mslp, col='red', lwd=2, lty='dashed')
    abline(h=thresh.slp, col='red', lwd=2, lty='dashed')
    
    # check exceedences independent
    Box.test(maxima$mslp[maxima$mslp >= thresh.mslp])
    Box.test(maxima$slp[maxima$slp >= thresh.slp])
  }
  
  # fit models...
  fit.mslp <- gpdAd(maxima$mslp[maxima$mslp >= thresh.mslp])
  maxima$thresh <- thresh.mslp
  scale <- fit.mslp$theta[1]
  shape <- fit.mslp$theta[2]
  maxima$scale <- scale
  maxima$shape <- shape
  maxima$p <- fit.mslp$p.value
  
  # semiparametric transform
  maxima$ecdf <- maxima$mslp
  maxima$ecdf[maxima$mslp < thresh.mslp] <- ecdf(maxima$mslp)(maxima$mslp[maxima$mslp < thresh.mslp])
  f.exceed <- (1 - ecdf(maxima$mslp)(thresh.mslp))
  maxima$ecdf[maxima$mslp >= thresh.mslp] <- 1 - f.exceed * pgpd(maxima$mslp[maxima$mslp >= thresh.mslp], thresh.mslp, scale, shape)
  slp.transformed.df <- rbind(slp.transformed.df, maxima)
}
rm(wind.df.all, slp.df.all)
transformed.df <- wind.transformed.df %>% inner_join(slp.transformed.df,
                                                     by = c('grid', 'cluster'),
                                                     suffix = c('.u10', '.slp'))
head(transformed.df)
# save csv to RELOAD in Python
transformed.df$erp <- 1 / (1 - transformed.df$extremeness.u10)
write.csv(transformed.df, paste0(indir, '/', 'fitted_data.csv'), row.names=FALSE)
