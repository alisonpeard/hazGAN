# Modification of texmex mexDependence and predict.mex to work with
# uniform marginals and skip GPD fitting.
library(texmex)


weibull_ecdf <- function(x) {
  n <- length(x)
  u <- ecdf(x)(x) * (n / (n + 1)) # convert to Weibull plotting positions
}

extcorr <- function(x, t=0.8) {
  n <- nrow(x)
  u <- weibull_ecdf(x[, 1])
  v <- weibull_ecdf(x[, 2])
  
  both_mask <- (u > t) & (v > t)
  if (sum(both_mask) < 3) {
    return(NA)
  }
  both_prob <- sum(both_mask) / n
  prob_u <- sum(u > t) / n
  chi <- both_prob / prob_u
  chi
}


metric_wrapper <- function(x, metric = "extcorr") {
  if (metric == "spearman") {
    cor(x[, 1], x[, 2], method = "spearman")
  } else if (metric == "pearson") {
    cor(x[, 1], x[, 2], method = "pearson")
  } else if (metric == "extcorr") {
    extcorr(x)
  }
}

define_margins <- function(margins) {
  margins <- list(
    casefold(margins),
    p2q = switch(casefold(margins),
                 gumbel = function(p) -log(-log(p)),
                 laplace = function(p) ifelse(
                   p < 0.5, log(2 * p),
                   -log(2 * (1 - p)))
    ),
    q2p = switch(casefold(margins),
                 gumbel = function(q) exp(-exp(-q)),
                 laplace = function(q) ifelse(
                   q < 0,
                   exp(q)/2, 1 - 0.5 * exp(-q)))
  )
  margins
}


fit_ht2004 <- function (
  u, which, dqu, margins = "gumbel", constrain = TRUE,
  v = 10, maxit = 1e+06) 
{
  # convert uniform margins to Gumbel/Laplace
  margins <- define_margins(margins)
  y <- margins$p2q(u)
  dth <- margins$p2q(dqu)

  # define dependent variable
  dependent <- (1:(dim(y)[[2]]))[-which]
  
  if (length(dqu) < length(dependent)) 
    dqu <- rep(dqu, length = length(dependent))
  
  aLow <- ifelse(
    margins[[1]] == "gumbel", # appropriate lower bound for parameter a
    10^(-10),
    -1 + 10^(-10)
    )

  start <- matrix(rep(c(0.01, 0.01), length(dependent)), nrow = 2)

  # This is the main optimisation function
  qfun <- function(
    X, yex, wh, aLow, margins, constrain, v, maxit, start
    ) {
    Qpos <- function(param, yex, ydep, constrain, v, aLow) {
      a <- param[1]
      b <- param[2]
      res <- texmex:::PosGumb.Laplace.negProfileLogLik(
        yex, ydep, a, b, constrain, v, aLow
        )
      res$profLik
    }
    
    o <- try(optim(par = start, fn = Qpos, control = list(maxit = maxit), 
                   yex = yex[wh], ydep = X[wh], constrain = constrain, 
                   v = v, aLow = aLow), silent = FALSE)
    
    if (o$par[1] <= 10^(-5) & o$par[2] < 0) { # for negative fits
      lo <- c(10^(-10), -Inf, -Inf, 10^(-10), -Inf, 10^(-10))
      
      Qneg <- function(yex, ydep, param) {
        param <- param[-1]
        b <- param[1]
        cee <- param[2]
        d <- param[3]
        m <- param[4]
        s <- param[5]
        
        obj <- function(yex, ydep, b, cee, d, m, s) {
          mu <- cee - d * log(yex) + m * yex^b
          sig <- s * yex^b
          log(sig) + 0.5 * ((ydep - mu)/sig)^2
        }
        res <- sum(obj(yex, ydep, b, cee, d, m, s))
        res
      }
      o <- try(optim(
        c(0, 0, 0, 0, 0, 1), 
        Qneg,
        method = "L-BFGS-B",
        lower = lo,
        upper = c(1, 1 - 10^(-10), Inf, 1 - 10^(-10), Inf, Inf),
        yex = yex[wh], ydep = X[wh]
        ), silent = FALSE)
      }
    else {
      Z <- (X[wh] - yex[wh] * o$par[1])/(yex[wh]^o$par[2])
      o$par <- c(o$par[1:2], 0, 0, mean(Z), sd(Z))
    }
    c(o$par[1:6], o$value)
  } # end of qfun
  
  yex <- c(y[, which])    # conditioning variable
  wh <- yex > unique(dth) # conditioning exceedances
  
  res <- sapply(
    1:length(dependent),
    function(X, dat, yex, wh, aLow, margins, constrain, v, maxit, start) qfun(dat[, X], yex, wh, aLow, margins, constrain, v, maxit, start[, X]),
    dat = as.matrix(y[, dependent]),
    yex = yex,
    wh = wh,
    aLow = aLow,
    margins = margins[[1]],
    constrain = constrain,
    v = v,
    maxit = maxit,
    start = start
    )
  
  loglik <- -res[7, ]
  res <- matrix(res[1:6, ], nrow = 6)
  dimnames(res)[[1]] <- c(letters[1:4], "m", "s") # a, b, c, f, m, s
  dimnames(res)[[2]] <- dimnames(u)[[2]][dependent] # dependent cols
  
  gdata <- as.matrix(y[wh, -which]) # put conditioning col first
  
  # Z = a = (Y - a(y)) / b(y) -> G(z) Eq.(3.3)
  tfun <- function(i, data, yex, a, b, cee, d) {
    data <- data[, i]
    a <- a[i] # location
    b <- b[i] # scale
    cee <- cee[i]
    d <- d[i]
    if (is.na(a)) 
      rep(NA, length(data))
    else {
      if (a < 10^(-5) & b < 0)  # Sect. 4.1: c = d = 0 unless a = 0 & b < 0
        a <- cee - d * log(yex)
      else a <- a * yex # Eq. (3.8)
      (data - a)/(yex^b)
    }
  }
  
  z <- try(sapply(
    1:(dim(gdata)[[2]]),
    tfun,
    data = gdata,
    yex = yex[wh],
    a = res[1, ],
    b = res[2, ],
    cee = res[3, ],
    d = res[4, ])
    )

  dimnames(z) <- list(NULL, dimnames(u)[[2]][dependent])
  
  # gather results in a list
  res2 <- list(
    coefficients = res,
    Z = z, # might not need this, we just need coefs
    dth = unique(dth),
    dqu = unique(dqu),
    which = which,
    conditioningVariable = colnames(u)[which],
    loglik = loglik,
    margins = margins,
    constrain = constrain,
    v = v
    )
  
  # return the margins and a list of the coefficients etc. (mexDependence)
  oldClass(res2) <- "mexDependence"
  output <- list(margins = u, dependence = res2)
  oldClass(output) <- "mex"
  output
}


predict_ht2004 <- function (object, which = 1, pqu = 0.99, nsim = 1000, trace = 10) 
{
  indata <- object$margins
  margins <- object$dependence$margins
  constrain <- object$dependence$constrain
  dall <- object
  
  MakeThrowData <- function(dcoef, z, data, pqu = 0.8) {
    ui <- runif(nsim, min = pqu) # simulating extremes of conditioning var
    y  <- margins$p2q(ui)
    z <- as.matrix(z[sample(1:(dim(z)[1]), size = nsim, replace = TRUE), ]) # what does this do??
    
    # make y_minus_i and z_minus_i vectors
    ymi <- sapply(1:(dim(z)[[2]]), makeYsubMinusI, z = z, v = dcoef, y = y)
    umi <- apply(ymi, 2, margins$q2p) # q2p
    
    sim <- data.frame(ui, umi)
    
    names(sim) <- c(
      colnames(data)[which],
      colnames(data)[-which]
    )
    
    # add mask for y where yi > ymi
    sim[, dim(sim)[2] + 1] <- y > apply(ymi, 1, max)
    sim
  }
  
  makeYsubMinusI <- function(i, z, v, y) {
    v <- v[, i]
    z <- z[, i]
    if (!is.na(v[1])) {
      if (v[1] < 10^(-5) & v[2] < 0) {
        if (v[4] < 10^(-5)) 
          d <- 0
        else d <- v[4]
        a <- v[3] - d * log(y)
      }
      else a <- v[1] * y
    }
    else a <- NA
    a + (y^v[2]) * z
  }
  
  sim <- MakeThrowData(
    dcoef = dall$dependence$coefficients, # fitted coeds (a,b,c,d,m,s)
    z = dall$dependence$Z, # what is this?
    data = indata # uniform margins
  )
  
  # only yi shoudl be the max?
  sim <- sim[, 1:(dim(sim)[2] - 1)] 
  
  data <- list(
    real = data.frame(indata[, which], indata[, -which]),
    simulated = data.frame(sim)
  )
  
  names(data$real)[1] <- colnames(indata)[which]
  
  res <- list(
    data = data,
    which = which,
    pqu = pqu
  )
  oldClass(res) <- "predict.mex"
  return(res)
}

