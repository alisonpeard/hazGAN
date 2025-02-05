# remember to select source on save
library(eva)
library(extRemes)
library(dplyr)
library(lubridate)
library(parallel)
library(future)
library(furrr)
library(data.table)

########### HELPER FUNCTIONS ###################################################
`%ni%` <- Negate(`%in%`)

res2str <- function(res){
  string <- paste0(res[1], "x", res[2])
  return(string)
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

scdf <- function(train, loc, scale, shape){
  # Add bounds checking for negative shape
  if (shape < 0) {
    upper_bound <- loc - scale/shape
    # Handle values beyond the upper bound
    beyond_bound <- x_tail >= upper_bound
    x_tail[beyond_bound] <- upper_bound - .Machine$double.eps
  }
  
  calculator <- function(x){
    u <- ecdf(train)(x)
    pthresh <- ecdf(train)(loc)
    tail_mask <- x > loc
    x_tail <- x[tail_mask]
    u_tail <- 1 - (1 - pthresh) * (1 - pgpd(x_tail, loc, scale, shape))
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

########### EVT FUNCTIONS ######################################################
standardise_by_month <- function(df, var) {
  
  df$month       <- months(df$time)
  df             <- df[,c(var, "month", "grid")]
  monthly_median <- aggregate(. ~ month + grid, df, median)
  
  df$monthly_median <- left_join(
    df[, c("month", "grid")],
    monthly_median,
    by = c("month" = "month", "grid" = "grid")
  )[[var]]
  
  df[[var]] <- df[[var]] - df$monthly_median

  return(list(
    var=df[[var]],
    medians=df[c("monthly_median", "grid", "month")]
    ))
}

# monthly_medians <- function(df, var) {
#   df <- df[, c(var, "time", "grid")]
#   df$month <- months(df$time)
#   monthly_median <- aggregate(. ~ month + grid,
#                               df[, c(var, "grid", "month")],
#                               median)
#   return(monthly_median)
# }

# remove_seasonality <- function(dt, vars) {
#   dt <- as.data.table(dt)
#   dt$month <- month(dt$time)
#   medians <- dt[, lapply(.SD, median), 
#                 by = .(month, grid), 
#                 .SDcols = vars]
#   
#   setkeyv(dt, c('month', 'grid'))
#   setkeyv(medians, c('month', 'grid'))
#   
#   for (var in vars) {
#     dt[medians, paste0(var, "_temp") := get(var) - get(paste0("i.", var))]
#   }
#   
#   standardised <- dt[, .SD, .SDcols = paste0(vars, "_temp")]
#   setnames(standardised, paste0(vars, "_temp"), vars)
#   
#   return(list(
#     standardised = as_tibble(standardised),
#     medians = medians
#   ))
# }


gridsearch <- function(series, var, qmin = 60, qmax = 99, rmin = 1, rmax = 14) {
  "Unit tests for this?"
  qvec <- c(qmin:qmax) / 100
  rvec <- c(rmin:rmax)

  print("Initial data summary:")
  print(summary(series[[var]]))

  nclusters <- matrix(nrow = length(rvec), ncol = length(qvec))
  ext_ind   <- matrix(nrow = length(rvec), ncol = length(qvec))
  pvals     <- matrix(nrow = length(rvec), ncol = length(qvec))

  series_var <- series[[var]]
  thresholds  <- quantile(series_var, qvec)

  print("Testing combinations:")
  for (i in seq_along(rvec)){
    for (j in seq_along(qvec)){
      thresh <- thresholds[j]

      d <- decluster(series_var, thresh = thresh,
                     r = rvec[i], method = "runs")

      # NOTE: theta = 1 a lot, double-check?
      e <- extremalindex(c(d), thresh, r = rvec[i],
                         method = "runs") # Coles (2001) §5.3.2

      # print(sprintf("r=%d, q=%.2f: ext_ind=%.3f",
      #               rvec[i], qvec[j], e[["extremal.index"]]))

      p <- Box.test(c(d)[c(d) > thresh], type = "Ljung")

      nclusters[i, j] <- e[["number.of.clusters"]]
      ext_ind[i, j]   <- e[["extremal.index"]]
      pvals[i, j]     <- p$p.value
    }
  }
  print("Before filtering:")
  print(table(is.finite(nclusters)))

  print("After extremal index filter:")
  nclusters[ext_ind < 0.8] <- -Inf # theta < 1 => extremal dependence
  print(table(is.finite(nclusters)))
  print("After p-value filter:")

  nclusters[pvals < 0.1]  <- -Inf  # H0: independent exceedances
  print(table(is.finite(nclusters)))

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

gpd_transformer <- function(df, metadata, var, q, chunksize=256) {
  gridcells <- unique(df$grid)
  df <- df[df$time %in% metadata$time, ]
  ngrid <- length(gridcells)
  
  # save df to RDS for worker access
  tmp <- tempfile(fileext = ".rds")
  saveRDS(df, tmp)
  rm(df)
  gc() 

  # chunk data for memory efficiency
  gridchunks <- split(gridcells, ceiling(seq_along(gridcells)/chunksize))
  gridchunks <- unname(gridchunks)
  nchunks <- length(gridchunks)

  # multiprocessing intiation
  plan(multisession, workers = min(availableCores() - 2, nchunks))
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
    excesses <- maxima$variable[maxima$variable >= thresh]
    p <- Box.test(excesses)[["p.value"]] # H0: independent
    if (p < 0.1) {
      warning(paste0(
        "p-value ≤ 10% for H0 of independent exceedences for gridcell ",
        grid_i, ". Value: ", round(p, 4)
      ))
    }

    # fit ECDF & GPD on train set only...
    maxima <- tryCatch({
      fit <- gpdAd(
        train$variable[train$variable >= thresh],
        bootstrap     = TRUE,
        bootnum       = 10,
        allowParallel = FALSE,
        numCores      = 1
      ) # H0: GPD distribution

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
