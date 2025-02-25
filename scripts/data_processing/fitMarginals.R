rm(list = ls())
library(arrow)

# set up env (depends if running or sourcing script)
try(setwd(getSrcDirectory(function(){})[1]))
try(setwd(dirname(rstudioapi::getActiveDocumentContext()$path)))

source("utils.R")
source("settings.R")

readRenviron("../../.env")

WD         <- Sys.getenv("TRAINDIR")  # nolint
WD         <- paste0(WD, "/", res2str(RES))
DRYRUN     <- TRUE
NDRYRUN    <- 10

daily    <- read_parquet(paste0(WD, "/", "daily.parquet"))
metadata <- read_parquet(paste0(WD, "/", "storms_metadata.parquet"))

# subset to mini dataset if it's a dry run
if(DRYRUN) {
  subgrid <- unique(daily$grid)[1:NDRYRUN]
  daily   <- daily[daily$grid %in% subgrid,]
}

#%%######## TRANSFORM STORMS ###################################################
# fit to marginal data
print("Tranforming fields...")
storms_wind <- weibull_transformer(daily, metadata, "u10", Q)
storms_mslp <- gpd_transformer(daily, metadata, "msl", Q)
storms_tp   <- gpd_transformer(daily, metadata, "tp", Q)

print("Done. Putting it all together...")
renamer <- function(df, var) {
  # rename all fields for joining dataframes
  df <- df %>%
    rename_with(~ paste0(., ".", var),
                -c("grid", "storm", "storm.rp", "variable"))
  df <- df %>% rename_with(~ var, "variable")
  return(df)
}

# storms_wind <- renamer(storms_wind, "u10")
storms_mslp <- renamer(storms_mslp, "mslp")
storms_tp   <- renamer(storms_tp, "tp")

storms <- storms_wind %>%
  inner_join(storms_mslp, by = c("grid", "storm", "storm.rp")) %>%
  inner_join(storms_tp, by = c("grid", "storm", "storm.rp"))

storms$thresh.q <- Q # keep track of threshold used

########### SAVE RESULTS #######################################################
if (!DRYRUN) {
  print("Saving...")
  write_parquet(storms, paste0(WD, "/", "storms.parquet"))
  cat("\nSaved as:", paste0(WD, "/", "storms.parquet"))
  print(paste0("Finished! ", length(unique(storms$storm)), " events processed."))
}
########### END ################################################################