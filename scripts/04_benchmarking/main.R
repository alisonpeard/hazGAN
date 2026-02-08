"Compare tail dependence between data, hazGAN, and Heffernan & Tawn (2004)"
# NOTE: results swap when I toggle refit_ecdf

library(ncdf4)
library(fields)
library(texmex)
library(eva)

source("utils.R")


# newer paths (Feb 2026)
data_path <- "/Users/alison/Documents/dphil/data/hazGAN/training/data.nc"
samples_path <- "/Users/alison/Documents/dphil/data/hazGAN/generated/rp10000/gaussian/nc/data.nc"

# how many pairs of points to sample
# 300 for dev, 1000 for report
n_pairs <- 1000
empirical <- TRUE
constrain_ht <- TRUE
thresh <- 0.9

ecdf <- function(x) {
  # ecdf that avoids (0, 1), Weibull plotting positions
  x <- sort(x)
  n <- length(x)
  if (n < 1) {
    stop("'x' must have 1 or more non-missing values")
  }
  vals <- unique(x)
  rval <- approxfun(vals, cumsum(tabulate(match(x, vals))) / (n + 1),
                    method = "constant",
                    rule = 2,
                    f = 0, ties = "ordered")
  class(rval) <- c("ecdf", "stepfun", class(rval))
  assign("nobs", n, envir = environment(rval))
  attr(rval, "call") <- sys.call()
  rval
}

extract_point <- function(data_array, loc_idx, refit_ecdf = empirical) {
  tmp <- data_array[, loc_idx, ] |> t() |> data.frame()
  colnames(tmp) <- c("u10", "tp", "mslp")
  
  if(refit_ecdf) {
    for(i in seq_along(colnames(tmp))) {
      tmp[, i] <- ecdf(tmp[, i])(tmp[, i])
    }
  }
  na.omit(tmp)
}

fit_spatial_model <- function(u1, u2) {

  u_u10 <- data.frame(cbind(u1[, 1], u2[, 1]))
  u_tp  <- data.frame(cbind(u1[, 2], u2[, 2]))
  u_mslp <- data.frame(cbind(u1[, 3], u2[, 3]))
  
  colnames(u_u10) <- c("i", "j")
  colnames(u_tp) <- c("i", "j")
  colnames(u_mslp) <- c("i", "j")
  
  u10  <- fit_dependence_model(u_u10, condition = 1)
  tp   <- fit_dependence_model(u_tp, condition = 1)
  mslp <- fit_dependence_model(u_mslp, condition = 1)  
  
  return(list(u10=u10, tp=tp, mslp=mslp))
}

fit_dependence_model <- function(u, condition = 1, qu = 0.8) {
  fit <- fit_ht2004(
    u, which = condition, dqu = qu, margins = "gumbel", constrain = constrain_ht
  )
  pred <- predict_ht2004(fit)
  pred$data$simulated[1:914, ]
}

# Load spatial grid
coords <- {
  nc <- nc_open(data_path)
  expand.grid(lon = ncvar_get(nc, "lon"), lat = ncvar_get(nc, "lat"))
}

# Mask to extreme storms for dependence modelling
mask <- {
  tmp <- nc_open(data_path) |> ncvar_get("anomaly")
  u10_data <- tmp[1, , , ]
  windmax <- apply(u10_data, 3, max, na.rm = TRUE)
  mask <- windmax > 15
  idx <- which(mask)
}

# load the training data on uniform scale
data <- {
  tmp <- nc_open(data_path) |> ncvar_get("uniform")
  tmp <- tmp[, , , mask, drop = FALSE]
  array(tmp, dim = c(dim(tmp)[1], nrow(coords), dim(tmp)[4]))
}

# load the hazGAN data on uniform scale
samples <- {
  tmp <- nc_open(samples_path) |> ncvar_get("uniform")
  array(tmp, dim = c(dim(tmp)[1], nrow(coords), dim(tmp)[4]))
}

# Initialize results storage
res <- data.frame(
  i = integer(n_pairs),
  j = integer(n_pairs),
  chi_base_u10_tp = numeric(n_pairs),
  chi_samp_u10_tp = numeric(n_pairs),
  chi_ht_u10_tp = numeric(n_pairs),
  chi_base_u10_mslp = numeric(n_pairs),
  chi_samp_u10_mslp = numeric(n_pairs),
  chi_ht_u10_mslp = numeric(n_pairs),
  chi_base_tp_mslp = numeric(n_pairs),
  chi_samp_tp_mslp = numeric(n_pairs),
  chi_ht_tp_mslp = numeric(n_pairs),
  chi_base_u10_spatial = numeric(n_pairs),
  chi_samp_u10_spatial = numeric(n_pairs),
  chi_ht_u10_spatial = numeric(n_pairs),
  chi_base_tp_spatial = numeric(n_pairs),
  chi_samp_tp_spatial = numeric(n_pairs),
  chi_ht_tp_spatial = numeric(n_pairs),
  chi_base_mslp_spatial = numeric(n_pairs),
  chi_samp_mslp_spatial = numeric(n_pairs),
  chi_ht_mslp_spatial = numeric(n_pairs)
)


set.seed(43)

# Main benchmarking loop
for (k in 1:n_pairs) {
  try({
    # Sample location indices
    i <- sample(nrow(coords), 1)
    j <- sample(nrow(coords), 1)
    
    # Extract data for locations i and j
    ui <- extract_point(data, i)
    uj <- extract_point(data, j)
    
    vi <- extract_point(samples, i)
    vj <- extract_point(samples, j)
    
    # Fit HT2004 models
    ui_ht     <- fit_dependence_model(ui)
    uiuj_ht   <- fit_spatial_model(ui, uj)
    uiuj_u10  <- uiuj_ht$u10
    uiuj_tp   <- uiuj_ht$tp
    uiuj_mslp <- uiuj_ht$mslp
    
    # Store indices
    res$i[k] <- i
    res$j[k] <- j
    
    # Multi-hazard dependence at location i
    res$chi_base_u10_tp[k] <- extcorr(ui[, 1:2], t=thresh)
    res$chi_samp_u10_tp[k] <- extcorr(vi[, 1:2], t=thresh)
    res$chi_ht_u10_tp[k]   <- extcorr(ui_ht[, 1:2], t=thresh)
    
    res$chi_base_u10_mslp[k] <- extcorr(ui[, c(1,3)], t=thresh)
    res$chi_samp_u10_mslp[k] <- extcorr(vi[, c(1,3)], t=thresh)
    res$chi_ht_u10_mslp[k]   <- extcorr(ui_ht[, c(1,3)], t=thresh)
    
    res$chi_base_tp_mslp[k] <- extcorr(ui[, 2:3], t=thresh)
    res$chi_samp_tp_mslp[k] <- extcorr(vi[, 2:3], t=thresh)
    res$chi_ht_tp_mslp[k]   <- extcorr(ui_ht[, 2:3], t=thresh)
    
    # Spatial dependence between locations i and j
    res$chi_base_u10_spatial[k] <- extcorr(cbind(ui[, 1], uj[, 1]), t=thresh)
    res$chi_samp_u10_spatial[k] <- extcorr(cbind(vi[, 1], vj[, 1]), t=thresh)
    res$chi_ht_u10_spatial[k] <- extcorr(uiuj_u10)
    
    res$chi_base_tp_spatial[k] <- extcorr(cbind(ui[,2], uj[,2]), t=thresh)
    res$chi_samp_tp_spatial[k] <- extcorr(cbind(vi[,2], vj[,2]), t=thresh)
    res$chi_ht_tp_spatial[k] <- extcorr(uiuj_tp)
    
    res$chi_base_mslp_spatial[k] <- extcorr(cbind(ui[,3], uj[,3]), t=thresh)
    res$chi_samp_mslp_spatial[k] <- extcorr(cbind(vi[,3], vj[,3]), t=thresh)
    res$chi_ht_mslp_spatial[k] <- extcorr(uiuj_mslp)
  })
  
  if (k %% 10 == 0) cat("Completed", k, "of", n_pairs, "iterations\n")
}

par(mfrow=c(3,2))
hist(ui[, 1]);hist(vi[, 1]);hist(ui[, 2]);hist(vi[, 2]);hist(ui[, 3]);hist(vi[, 3]);
# note the left-skew of the subset data in the parent-uniform domain
# this implies that extracting samples by the domain-global maximum is
# effectively 'lifting' the mass of the distribution; this implies that the
# data has some association in the extremes.

# Save results as a dataframe
write.csv(res, "extcorrs.csv", row.names = FALSE)
cat("res saved to extcorrs.csv\n")