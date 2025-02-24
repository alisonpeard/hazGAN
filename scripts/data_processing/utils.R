# remember to select source on save
library(zoo)
library(eva)
library(extRemes)
library(dplyr)
library(lubridate)
library(parallel)
library(future)
library(furrr)
library(data.table)
library(progress)  # Add this
library(magrittr)  # Optional but recommended for pipe operator
library(stats)

########### HELPER FUNCTIONS ###################################################
`%ni%` <- Negate(`%in%`)

res2str <- function(res){
  string <- paste0(res[1], "x", res[2])
  return(string)
}
standardise_by_month <- function(df, var) {
  df$month <- months(df$time)
  df <- df[,c(var, "month", "grid")]
  monthly_median <- aggregate(. ~ month + grid, df, median)
  df$monthly_median <- left_join(
    df[, c("month", "grid")],
    monthly_median,
    by = c("month" = "month", "grid" = "grid")
  )[[var]]
  df[[var]] <- df[[var]] - df$monthly_median
  return(df[[var]])
}
monthly_medians <- function(df, var) {
  df <- df[, c(var, "time", "grid")]
  df$month <- months(df$time)
  monthly_median <- aggregate(. ~ month + grid,
                              df[, c(var, "grid", "month")],
                              median)
  return(monthly_median)
}
ecdf <- function(x) {
  x <- sort(x)
  n <- length(x)
  if (n < 1) {
    stop("'x' must have 1 or more non-missing values")
  }
  vals <- unique(x)
  rval <- approxfun(vals, cumsum(tabulate(match(x, vals))) / (n + 1),
                    method = "constant", #yleft = 0, yright = 1,
                    rule = 2, # take values at extremes
                    f = 0, ties = "ordered")
  class(rval) <- c("ecdf", "stepfun", class(rval))
  assign("nobs", n, envir = environment(rval))
  attr(rval, "call") <- sys.call()
  rval
}
scdf <- function(train, loc, scale, shape, cdf = pgd){
  # Note, trialing using excesses and setting loc=0
  # This is for flexibility with cdf choice
  calculator <- function(x){
    u <- ecdf(train)(x)
    pthresh <- ecdf(train)(loc)
    tail_mask <- x > loc
    x_tail <- x[tail_mask]
    exceedances <- x_tail - loc
    u_tail <- 1 - (1 - pthresh) * (1 - cdf(exceedances, scale=scale, shape=shape))
    u[tail_mask] <- u_tail
    return(u)
  }
  return(calculator)
}
progress_bar <- function(n, prefix = "", suffix = "") {
  pb <- utils::txtProgressBar(min = 0, max = n, style = 3)
  function(i) {
    utils::setTxtProgressBar(pb, i)
    if (i == n) close(pb)
  }
}

########### EVT FUNCTIONS ###################################################
gridsearch <- function(series, var, qmin = 60, qmax = 99, rmax = 14) {
  "Unit tests for this?"
  qvec <- c(qmin:qmax) / 100
  rvec <- c(1:rmax)

  nclusters <- matrix(nrow = length(rvec), ncol = length(qvec))
  ext_ind <- matrix(nrow = length(rvec), ncol = length(qvec))
  pvals <- matrix(nrow = length(rvec), ncol = length(qvec))

  for (i in seq_along(rvec)){
    for (j in seq_along(qvec)){
      thresh <- quantile(series[[var]], qvec[j])
      d <- decluster(series[[var]], thresh = thresh,
                     r = rvec[i], method = "runs")

      # NOTE: theta = 1 a lot, double-check?
      e <- extremalindex(c(d), thresh, r = rvec[i],
                         method = "runs") # Coles (2001) §5.3.2
      p <- Box.test(c(d)[c(d) > thresh], type = "Ljung")
      nclusters[i, j] <- e[["number.of.clusters"]]
      ext_ind[i, j] <- e[["extremal.index"]]
      pvals[i, j] <- p$p.value
    }
  }

  # remove any cases with dependence in exceedences
  nclusters[ext_ind < 0.8] <- -Inf # theta < 1 => extremal dependence
  nclusters[pvals < 0.1]  <- -Inf  # H0: independent exceedances
  ind <- which(nclusters == max(nclusters), arr.ind = TRUE)
  r <- rvec[ind[1]]
  q <- qvec[ind[2]]
  p <- pvals[ind[1], ind[2]]
  return(list(r = r, q = q, p = p))
}
storm_extractor <- function(daily, var, rfunc) {
  series <- aggregate(. ~ time, daily[, c("time", var)], rfunc)

  # gridsearch run lengths and thresholds
  result <- gridsearch(series, var)
  r <- result$r
  q <- result$q
  p <- result$p

  thresh <- quantile(series[[var]], q)
  cat(paste0(
    "Final selection from gridsearch:\n",
    "Run length: ", r, "\n",
    "Quantile: : ", q, "\n",
    "Threshold: ", round(thresh, 4), "\n",
    "P-value (H0:independent): ", round(p, 4), "\n"
  ))

  # final declustering
  declustering <- decluster(series[[var]], thresh, r = r)
  storms <- attr(declustering, "clusters")
  times <- series$time[series[[var]] > thresh]
  variable <- series[[var]][series[[var]] > thresh]
  metadata <- data.frame(time = times, storm = storms, variable = variable)

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

  # Ljung-box again
  p <- Box.test(c(storms$variable), type = "Ljung")$p.value
  cat(paste0("Final Ljung-Box p-value: ", round(p, 4), '\n'))

  # storm frequency
  m <- nrow(storms)
  nyears <- length(unique(year(daily$time)))
  lambda <- m / nyears
  metadata$lambda <- lambda
  cat(paste0("Number of storms: ", m, '\n'))

  # assign return periods
  survival_prob <- 1 - (
    rank(storms$variable, ties.method = "average") / (m + 1)
  )
  rp <- 1 / (lambda * survival_prob)
  storms$storm.rp <- rp

  # remaining metadata
  metadata <- left_join(metadata,
                        storms[c("storm", "storm.rp", "storm.size")],
                        by = c("storm"))
  metadata <- metadata %>% rename_with(~ var, variable)

  return(metadata)
}
select_gpd_threshold <- function(var, min_exceedances = 20,
                                 nthresholds = 50, nsim = 20) {
  thresholds <- quantile(var, probs = seq(0.8, 0.98, length.out=nthresholds))

  fits <- gpdSeqTests(var, thresholds = thresholds, method="ad", nsim=nsim)

  valid_n          <- fits$num.above >= min_exceedances
  valid_stops      <- fits$ForwardStop < 1.5

  shape_changes    <- abs(diff(fits$est.shape))
  shape_stability  <- c(
    TRUE,
    shape_changes < quantile(
      shape_changes, 0.95, na.rm = TRUE
    )
  )

  valid_thresholds <- which(valid_n & valid_stops & shape_stability)

  if (length(valid_thresholds) > 0) {
    shape_volatility <- rollapply(fits$est.shape[valid_thresholds],
                                  width = 3,
                                  FUN = function(x) diff(range(x)),
                                  fill = NA)

    idx <- valid_thresholds[which.min(shape_volatility)]
    return(list(
      thresh   = fits$threshold[idx],
      theta    = c(fits$est.scale[idx], fits$est.shape[idx]),
      p.value  = fits$p.values[idx],
      n_exceed = fits$num.above[idx]
    ))
  } else {
    return(NULL)
  }
}
gpd_transformer <- function(df, metadata, var, q, chunksize = 256) {
  gridcells <- unique(df$grid)
  df <- df[df$time %in% metadata$time, ]
  ngrid <- length(gridcells)
  
  # save df to RDS for worker access
  tmp <- tempfile(fileext = ".rds")
  saveRDS(df, tmp)
  rm(df)
  gc() 

  # chunk data for memory efficiency
  gridchunks <- split(gridcells, ceiling(seq_along(gridcells) / chunksize))
  gridchunks <- unname(gridchunks)
  nchunks <- length(gridchunks)

  # multiprocessing intiation
  plan(multisession, workers = min(availableCores() - 4, nchunks))
  pb <- progress::progress_bar$new(
    format = "Processing grid cells [:bar] :percent eta: :eta",
    total  = ngrid
  )

  # main GPD fitting function
  process_gridcell <- function(grid_i, df) {
    gridcell <- df[df$grid == grid_i, ]
    gridcell <- left_join(gridcell,
                          metadata[, c("time", "storm", "storm.rp")],
                          by = c("time" = "time"))
    
    maxima <- gridcell %>%
      group_by(storm) %>%
      slice(which.max(get(var))) %>%
      summarise(
        variable = max(get(var)),
        time = time,
        storm.rp = storm.rp,
        grid = grid
      )
    
    train <- maxima[year(maxima$time) %ni% TEST.YEARS,]
    thresh <- quantile(train$variable, q)

    # validation
    excesses <- maxima$variable[maxima$variable > thresh]
    p <- Box.test(excesses)[["p.value"]] # H0: independent
    if (p < 0.1) {
      warning(paste0(
        "p-value ≤ 10% for H0 of independent exceedences for gridcell ",
        grid_i, ". Value: ", round(p, 4)
      ))
    }

    # fit ECDF & GPD on train set only...
    maxima <- tryCatch({
      # fit <- gpdAd(
      #   train$variable[train$variable > thresh],
      #   bootstrap     = TRUE,
      #   bootnum       = 10,
      #   allowParallel = FALSE,
      #   numCores      = 1
      # ) # H0: GPD distribution
      
      fit <- select_gpd_threshold(train$variable)
      thresh <- fit$thresh
      scale <- fit$theta[1]
      shape <- fit$theta[2]
      maxima$thresh <- thresh
      maxima$scale  <- scale
      maxima$shape  <- shape
      maxima$p      <- fit$p.value

      # empirical cdf transform
      maxima$scdf <- scdf(train$variable, thresh, scale, shape)(maxima$variable)
      maxima$ecdf <- ecdf(train$variable)(maxima$variable)
      maxima
    }, error = function(e) {
      warning(sprintf("MLE failed for grid cell %d: %s. Resorting to fully empirical fits.",
                      grid_i, e$message))
      maxima$thresh <- NA
      maxima$scale  <- NA
      maxima$shape  <- NA
      maxima$p      <- 0
      maxima$ecdf <- ecdf(train$variable)(maxima$variable)
      maxima$scdf <- maxima$ecdf
      maxima
    })
    #pb$tick()
    return(maxima)
  }
  
  
  # wrapper for process_gridcell()
  process_gridchunk <- function(gridchunk) {
    df <- readRDS(tmp)
    df <- df[df$grid %in% gridchunk, ]
    gc()
    
    maxima <- lapply(gridchunk, function(grid_i) {
      process_gridcell(grid_i, df)
    })
    
    bind_rows(maxima)
  }
  
  # apply multiprocessing
  transformed <- future_map_dfr(
    .x = gridchunks,
    .f = process_gridchunk,
    .options = furrr_options(
      seed = TRUE,
      scheduling = 1
    )
  )
  
  unlink(tmp)
  
  fields <- c("storm", "variable", "time", "storm.rp",
              "grid", "thresh", "scale", "shape", "p",
              "ecdf", "scdf")
  transformed <- transformed[, fields]
  
  return(transformed)
}
select_weibull_threshold <- function(var, thresholds,
                                     method = "mle", nsim = nsim) {
  loglikelihood <- function(params, data) {
    shape <- params[1]
    scale <- params[2]
    
    # Negative log-likelihood for Weibull
    -sum(dweibull(data, shape=shape, scale=scale, log=TRUE))
  }
  
  ks.gof <- function(x, shape, scale) {
    result <- ks.test(x, pweibull, shape=shape, scale=scale)
    result
  }
  
  shapes <- vector(length=length(thresholds))
  scales <- vector(length=length(thresholds))
  likelihoods <- vector(length=length(thresholds))
  n_exceeds <- vector(length=length(thresholds))
  
  for (i in 1:length(thresholds)) {
    q <- thresholds[i]
    exceedances <- var[var > q] - q
    
    # initial estimates
    mean_exc <- mean(exceedances)
    var_exc <- var(exceedances)
    init_shape <- 2  # like Rayleigh distribution for winds
    init_scale <- mean_exc
    
    # fit MLE
    fit <- optim(c(init_shape, init_scale), 
                        loglikelihood, 
                        data=exceedances,
                        method="L-BFGS-B",
                        lower=c(0.1, 0.1),
                        upper=c(2, 10))
    
    shapes[i]      <- fit$par[1]
    scales[i]      <- fit$par[2]
    likelihood[i] <- fit$value
    n_exceeds[i] <- length(exceedances)
  }

  best <- which.min(likelihood)
  
  thresh <- thresholds[best]
  shape <- shapes[best]
  scale <- scales[best]
  likelihood <- likelihoods[best]
  
  exceedances <- x[x > thresh] - thresh
  n_exceed <- length(exceedances)
  pval <- ks.gof(exceedances, shape, scale)$p.value
  
  return(list(
    thresh=thresh,
    thetha=c(scale, shape),
    p.value=pval,
    n_exceed=n_exceed
    ))
}
weibull_transformer <- function(df, metadata, var, q, chunksize = 256) {
  gridcells <- unique(df$grid)
  df <- df[df$time %in% metadata$time, ]
  ngrid <- length(gridcells)
  
  # save df to RDS for worker access
  tmp <- tempfile(fileext = ".rds")
  saveRDS(df, tmp)
  rm(df)
  gc() 
  
  # chunk data for memory efficiency
  gridchunks <- split(gridcells, ceiling(seq_along(gridcells) / chunksize))
  gridchunks <- unname(gridchunks)
  nchunks <- length(gridchunks)
  
  # multiprocessing initiation
  plan(multisession, workers = min(availableCores() - 4, nchunks))
  pb <- progress::progress_bar$new(
    format = "Processing grid cells [:bar] :percent eta: :eta",
    total  = ngrid
  )
  
  # main GPD fitting function
  process_gridcell <- function(grid_i, df) {
    gridcell <- df[df$grid == grid_i, ]
    gridcell <- left_join(gridcell,
                          metadata[, c("time", "storm", "storm.rp")],
                          by = c("time" = "time"))
    
    maxima <- gridcell %>%
      group_by(storm) %>%
      slice(which.max(get(var))) %>%
      summarise(
        variable = max(get(var)),
        time = time,
        storm.rp = storm.rp,
        grid = grid
      )
    
    train <- maxima[year(maxima$time) %ni% TEST.YEARS,]
    thresh <- quantile(train$variable, q)
    
    # validation
    excesses <- maxima$variable[maxima$variable > thresh]
    p <- Box.test(excesses)[["p.value"]] # H0: independent
    if (p < 0.1) {
      warning(paste0(
        "p-value ≤ 10% for H0 of independent exceedences for gridcell ",
        grid_i, ". Value: ", round(p, 4)
      ))
    }
    
    # fit ECDF & GPD on train set only...
    maxima <- tryCatch({
      fit <- select_weibull_threshold(train$variable)
      thresh <- fit$thresh
      scale  <- fit$theta[1]
      shape  <- fit$theta[2]
      pval   <- fit$p.value
      maxima$thresh <- thresh
      maxima$scale  <- scale
      maxima$shape  <- shape
      maxima$p      <- pval
      
      # empirical cdf transform
      maxima$scdf <- scdf(train$variable, thresh, scale, shape,
                          cdf = pweibull)(maxima$variable) # TO DO!
      
      maxima$ecdf <- ecdf(train$variable)(maxima$variable)
      maxima
    }, error = function(e) {
      warning(sprintf("MLE failed for grid cell %d: %s. Resorting to fully empirical fits.",
                      grid_i, e$message))
      maxima$thresh <- NA
      maxima$scale  <- NA
      maxima$shape  <- NA
      maxima$p      <- 0
      maxima$ecdf <- ecdf(train$variable)(maxima$variable)
      maxima$scdf <- maxima$ecdf
      maxima
    })
    return(maxima)
  }
  
  
  # wrapper for process_gridcell()
  process_gridchunk <- function(gridchunk) {
    df <- readRDS(tmp)
    df <- df[df$grid %in% gridchunk, ]
    gc()
    
    maxima <- lapply(gridchunk, function(grid_i) {
      process_gridcell(grid_i, df)
    })
    
    bind_rows(maxima)
  }
  
  # apply multiprocessing
  transformed <- future_map_dfr(
    .x = gridchunks,
    .f = process_gridchunk,
    .options = furrr_options(
      seed = TRUE,
      scheduling = 1
    )
  )
  
  unlink(tmp)
  
  fields <- c("storm", "variable", "time", "storm.rp",
              "grid", "thresh", "scale", "shape", "p",
              "ecdf", "scdf")
  transformed <- transformed[, fields]
  print("I just transformed a marginal using a Weibull PIT!")
  return(transformed)
}

# # OLD
# gpd_transformer <- function(df, metadata, var, q) {
#   gridcells <- unique(df$grid)
#   ngrid <- length(gridcells)

#   update_progress <- progress_bar(ngrid, "Fitting GPD to excesses:", "Complete")

#   df <- df[df$time %in% metadata$time, ]

#   fields <- c("storm", "variable", "time", "storm.rp",
#               "grid", "thresh", "scale", "shape", "p", "ecdf")
#   transformed <- data.frame(matrix(nrow = 0, ncol = length(fields)))
#   colnames(transformed) <- fields

#   ncores  <- min(detectCores(), ngrid)
#   cluster <- makeCluster(ncores)
#   clusterExport(cluster, c("df", "var", "q", "TEST.YEARS", "gpdAd", "scdf", "ecdf"))

#   progress_file <- tempfile()
#   writeLines("0", progress_file)

#   transformed <- parLapply(cluster, 1:ngrid, function(i) {
#   # for (i in 1:ngrid){
#     grid_i <- gridcells[i]
#     gridcell <- df[df$grid == grid_i, ]
#     gridcell <- left_join(gridcell,
#                           metadata[, c("time", "storm", "storm.rp")],
#                           by = c("time" = "time"))
#     maxima <- gridcell %>%
#       group_by(storm) %>%
#       slice(which.max(get(var))) %>%
#       summarise(
#         variable = max(get(var)),
#         time = time,
#         storm.rp = storm.rp,
#         grid = grid
#       )
    
#     train <- maxima[year(maxima$time) %ni% TEST.YEARS,]
#     thresh <- quantile(train$variable, q)

#     # validation
#     excesses <- maxima$variable[maxima$variable > thresh]
#     p <- Box.test(excesses)[["p.value"]] # H0: independent
#     if (p < 0.1) {
#       warning(paste0(
#         "p-value ≤ 10% for H0 of independent exceedences for gridcell ",
#         i, ". Value: ", round(p, 4)
#       ))
#     }

#     # fit ECDF & GPD on train set only...
#     newrow <- tryCatch({
#       fit <- gpdAd(
#         train$variable[train$variable > thresh],
#         bootstrap     = TRUE,
#         bootnum       = 10,
#         allowParallel = FALSE,
#         numCores      = 2
#       ) # H0: GPD distribution

#       scale <- fit$theta[1]
#       shape <- fit$theta[2]
#       maxima$thresh <- thresh
#       maxima$scale  <- scale
#       maxima$shape  <- shape
#       maxima$p      <- fit$p.value

#       # empirical cdf transform
#       maxima$scdf <- scdf(train$variable, thresh,
#                           scale, shape)(maxima$variable)
#       maxima$ecdf <- ecdf(train$variable)(maxima$variable)

#       maxima # assigns maxima to newrow
#     }, error = function(e) {
#       print(paste0("MLE failed for grid cell ", grid_i, " ", e))
#       maxima$thresh <- NA
#       maxima$scale  <- NA
#       maxima$shape  <- NA
#       maxima$p      <- 0
#       maxima$ecdf <- ecdf(train$variable)(maxima$variable)
#       maxima$scdf <- maxima$ecdf
#       return(maxima)
#     })

#     # update progress
#     progress <- as.integer(readLines(progress_file)[1])
#     writeLines(as.character(progress + 1), progress_file)
#     update_progress(as.integer(readLines(progress_file)[1]))

#     return(newrow)
#   })

#   stopCluster(cluster)
#   unlink(progress_file)
#   transformed <- do.call(rbind, transformed)
  
#   return(transformed)
# }
