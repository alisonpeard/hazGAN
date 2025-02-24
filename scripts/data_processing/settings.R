FILENAME   <- "data_1941_2022.nc"     # nolint
RES        <- c(64, 64)               # nolint
RFUNC      <- max                     # nolint, https://doi.org/10.1111/rssb.12498
TEST.YEARS <- c(2022)                 # nolint, exclude from ecdf + gpd fitting
VISUALS    <- TRUE                    # nolint
Q          <- 0.95                    # nolint