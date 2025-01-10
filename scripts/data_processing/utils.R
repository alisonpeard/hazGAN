# remember to select source on save
library(eva)
library(extRemes)
library(dplyr)
library(lubridate)
library(parallel)

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
scdf <- function(train, loc, scale, shape){
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


gpd_transformer <- function(df, metadata, var, q) {
  gridcells <- unique(df$grid)
  ngrid <- length(gridcells)

  update_progress <- progress_bar(ngrid, "Fitting GPD to excesses:", "Complete")

  df <- df[df$time %in% metadata$time, ]

  fields <- c("storm", "variable", "time", "storm.rp",
              "grid", "thresh", "scale", "shape", "p", "ecdf")
  transformed <- data.frame(matrix(nrow = 0, ncol = length(fields)))
  colnames(transformed) <- fields

  ncores  <- min(detectCores(), ngrid)
  cluster <- makeCluster(ncores)
  clusterExport(cluster, c("df", "var", "q", "TEST.YEARS", "gpdAd", "scdf", "ecdf"))

  progress_file <- tempfile()
  writeLines("0", progress_file)

  transformed <- parLapply(cluster, 1:ngrid, function(i) {
  # for (i in 1:ngrid){
    grid_i <- gridcells[i]
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
        i, ". Value: ", round(p, 4)
      ))
    }

    # fit ECDF & GPD on train set only...
    newrow <- tryCatch({
      fit <- gpdAd(
        train$variable[train$variable >= thresh],
        bootstrap     = TRUE,
        bootnum       = 10,
        allowParallel = FALSE,
        numCores      = 2
      ) # H0: GPD distribution

      scale <- fit$theta[1]
      shape <- fit$theta[2]
      maxima$thresh <- thresh
      maxima$scale  <- scale
      maxima$shape  <- shape
      maxima$p      <- fit$p.value

      # empirical cdf transform
      maxima$scdf <- scdf(train$variable, thresh,
                          scale, shape)(maxima$variable)
      maxima$ecdf <- ecdf(train$variable)(maxima$variable)

      maxima # assigns maxima to newrow
    }, error = function(e) {
      print(paste0("MLE failed for grid cell ", grid_i, " ", e))
      maxima$thresh <- NA
      maxima$scale  <- NA
      maxima$shape  <- NA
      maxima$p      <- 0
      maxima$ecdf <- ecdf(train$variable)(maxima$variable)
      maxima$scdf <- maxima$ecdf
      return(maxima)
    })

    # update progress
    progress <- as.integer(readLines(progress_file)[1])
    writeLines(as.character(progress + 1), progress_file)
    update_progress(as.integer(readLines(progress_file)[1]))

    return(newrow)
  })

  stopCluster(cluster)
  unlink(progress_file)
  transformed <- do.call(rbind, transformed)
  
  return(transformed)
}
