# exploring distributions
library(eva)
library(ggsci)
#library(optim)
source("utils.R") # better ecdf

n <- 1000
q <- 0.9
k0 <- 0.5
xweibull <- rweibull(n, k0)
threshold <- quantile(xweibull, q)
hist(xweibull)

tail <- xweibull[xweibull > threshold]
tail <- sort(tail)

N1 <- function(k) {
  parametric <- (1 - pweibull(tail, shape=k)) / (1 - pweibull(threshold, shape=k))
  empirical <- 1- ecdf(tail)(tail)
  return(mean(((parametric / empirical) - 1)**2))
}

result <- optim(par=0.1,
                fn=N1,
                method="L-BFGS-B",
                lower=1e-6,
                upper=Inf)


# view results
k <- result$par

colors <- pal_npg()(2) #terrain.colors(6)

par(mfrow=c(1, 2))

parametric <- (1 - pweibull(tail, shape=k0)) / (1 - pweibull(threshold, shape=k0))
empirical <- 1- ecdf(tail)(tail)
plot(tail, parametric, col=colors[1], type="l", lwd=3, main="With true k")
lines(tail, empirical, col=colors[2], lwd=3)
legend("topright", 
       legend=c("Parametric", "Empirical"),
       col=colors,
       lty=1,
       lwd=3)

parametric <- (1 - pweibull(tail, shape=k)) / (1 - pweibull(threshold, shape=k))
empirical <- 1- ecdf(tail)(tail)
colors <- pal_npg()(2) #terrain.colors(6)
plot(tail, parametric, col=colors[1], type="l", lwd=3, main="With estimated k")
lines(tail, empirical, col=colors[2], lwd=3)
legend("topright", 
       legend=c("Parametric", "Empirical"),
       col=colors,
       lty=1,
       lwd=3)



