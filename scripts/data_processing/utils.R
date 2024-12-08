########### HELPER FUNCTIONS ###################################################
`%ni%` <- Negate(`%in%`)
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
  rankings <- rank(x, ties.method=c("average"))
  n <- length(x)
  return(rankings/(n+1))
}
progress_bar <- function(n, prefix="", suffix="") {
  pb <- utils::txtProgressBar(min = 0, max = n, style = 3)
  function(i) {
    utils::setTxtProgressBar(pb, i)
    if (i == n) close(pb)
  }
}

########### EVT FUNCTIONS ###################################################
storm.extractor <- function(daily, var, rfunc){
  "Possibly divide this into smaller functions later."
  series <- aggregate(. ~ time, daily[,c('time', var)], rfunc)
  
  qvec <- c(60:99) / 100
  rvec <- c(1:14)
  nclusters <- matrix(nrow=length(rs), ncol=length(qs))
  ext.ind <- matrix(nrow=length(rs), ncol=length(qs))
  p.vals <- matrix(nrow=length(rs), ncol=length(qs))
  
  for(i in 1:length(rs)){
    for(j in 1:length(qs)){
      thresh <- quantile(series[[var]], qs[j])
      d <- decluster(series[[var]], thresh=thresh, r=rs[i], method='runs')
      # NOTE: θ=1 a lot, double-check?
      e <- extremalindex(c(d), thresh, r=rs[i], method='runs') 
      p <- Box.test(c(d)[c(d) > thresh], type='Ljung')
      nclusters[i, j] <- e[['number.of.clusters']]
      ext.ind[i,j] <- e[['extremal.index']]
      p.vals[i, j] <- p$p.value
    }
  }
  
  # remove any cases with dependence in exceedences
  nclusters[ext.ind < 0.8] <- -Inf # theta < 1 => dependency
  nclusters[p.vals < 0.1] <- -Inf  #  
  ind <- which(nclusters == max(nclusters), arr.ind=TRUE)
  r <- rs[ind[1]]
  q <- qs[ind[2]]
  
  # make data.frame of events
  thresh <- quantile(series[[var]], q)
  print("Selection (r, quantile, threshold):")
  print(paste(r, q, round(thresh, 4), sep=", "))
  
  # final declustering
  declustering <- decluster(series[[var]], thresh, r=r)
  storms <- attr(declustering, 'clusters')
  times <- series$time[series[[var]] > thresh] # times when max(u10) exceeds thresh
  variable <- series[[var]][series[[var]] > thresh]
  metadata <- data.frame(time=times, storm=storms, variable=variable)
  
  # storm stats
  storms <- metadata %>%
    group_by(storm) %>%
    mutate(storm.size = n()) %>%
    slice(which.max(variable)) %>%
    summarise(
      variable = max(variable),
      time = time,
      storm.size = storm.size
      )
  
  # empirical return period calculation
  m <- nrow(storms); paste0("Number of storms: ", m)
  nyears <- length(unique(year(daily$time)))
  lambda <- m / nyears; paste0("Yearly rate: ", lambda) # yearly occurrence rate
  metadata$lambda <- lambda
  survival.prob <- 1 - (rank(storms$variable, ties.method='average')) / (m + 1)
  rp <- 1 / (lambda*survival.prob)
  storms$storm.rp <- rp
  
  metadata <- left_join(metadata, storms[c('storm', 'storm.rp', 'storm.size')], by=c('storm'))
  metadata$q <- q
  
  print('Declustered dataframe:')
  head(metadata[order(-metadata$storm.rp),])
  
  metadata <- metadata %>% rename_with(~ var, variable)
  
  return(metadata)
}

gpd.transformer <- function(df, VAR, q){
  update_progress <- progress_bar(ngrid, "Fitting GPD to excesses:", "Complete")
  fields <- c("storm", "variable", "time", "storm.rp", "grid", "thresh", "scale", "shape", "p", "ecdf")
  transformed <- data.frame(matrix(nrow=0, ncol=length(fields)))
  colnames(transformed) <- fields
  
  for(i in 1:ngrid){
    GRIDCELL <- grid.cells[i]
    gridcell <- df[df$grid == GRIDCELL,]
    gridcell <- left_join(gridcell, cluster.df[,c('time', 'storm', 'storm.rp')], by=c('time'='time'))
    
    maxima <- gridcell %>%
      group_by(storm) %>%
      slice(which.max(get(VAR))) %>%
      summarise(
        variable = max(get(VAR)),
        time = time,
        storm.rp = storm.rp,
        grid = grid
      )
    thresh <- quantile(maxima$variable, q)
    
    # validation
    excesses <- maxima$variable[maxima$variable >= thresh]
    p <- Box.test(excesses)[['p.value']] # H0: independent # print output if verbose
    if(p < 0.1){
      warning(paste0("p-value <= 10% for H0 of independent exceedences for gridcell ", i, ". Value: ", round(p, 4)))
    }
    
    # fit models...
    new.row <- tryCatch({
      fit <- gpdAd(
        maxima$variable[maxima$variable >= thresh],
        bootstrap = TRUE,
        bootnum = 10,
        allowParallel = TRUE,
        numCores = 2
      ) # H0: GPD distribution
      
      scale <- fit$theta[1]
      shape <- fit$theta[2]
      maxima$thresh <- thresh
      maxima$scale  <- scale
      maxima$shape  <- shape
      maxima$p      <- fit$p.value
      
      # empirical cdf transform (new 03-09-2024)
      maxima$ecdf <- get.ecdf(maxima$variable)
      
      maxima # assigns maxima to new.row
    }, error=function(e){
      print(paste0("skipping grid cell ", GRIDCELL, ' ', e))
      missing.columns <- setdiff(names(transformed), names(maxima))
      maxima[missing.columns] <- NA
      return(maxima)
    })
    transformed <- rbind(transformed, new.row)
    update_progress(i)
  }
  return(transformed)
}



