# copying /Users/alison/Documents/DPhil/multivariate/_archive/evtGAN_code&data_download04082023/Maxima_rain/evt_rain.R
# Example from vignette
rm(rs = list())
library(SpatialExtremes)

## 1. Smith's model
set.seed(8)
x <- y <- seq(0, 10, length = 100)
coord <- cbind(x, y)
data <- rmaxstab(1, coord, "gauss", cov11 = 9/8, cov12 = 0, cov22 = 9/8,
                 grid = TRUE)
##We change to unit Gumbel margins for visibility
filled.contour(x, y, log(data), color.palette = terrain.colors)

## 2. Schlather's model
data <- rmaxstab(1, coord, cov.mod = "powexp", nugget = 0, range = 3,
                 smooth = 1, grid = TRUE)
filled.contour(x, y, log(data), color.palette = terrain.colors)

# Define the coordinate of each location
n.site <- 30
locations <- matrix(runif(2 * n.site, 0, 10), ncol = 2)
colnames(locations) <- c("lon", "lat")

## 3. Brown--Resnick's model **** Only available for non gridded points currently ****
data <- rmaxstab(1, coord, cov.mod = "brown", range = 3, smooth = 0.5, grid=TRUE)
filled.contour(x, y, data, color.palette = terrain.colors)
