"
----Code for model selection----
Adapted from Yu Mo's code for [cite paper].
"
rm(list = ls())
library(parallel)
library(dplyr)
library(tidyr)
library(parallel)

wd <- '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/'
use.models <- c('xgb') # c('rf', 'gbm', 'svm', 'knn', 'xgb', 'rpart')
src <- 'final'
out <- paste0(src, '_', paste(use.models, collapse='_'))
infile <- paste0(wd, src, '.csv')
outfile <- paste0(wd, out, '.csv')
outfig <- paste0(wd, out, '.pdf')
                 
allYs <- c("intensity")
allXs <- c(
  # "elevation_mean",
  # "landingPressure"
  # "landingLat",
  # "landingLon"
  #"era5_wind",
  #"era5_precip"
  "landingWindMaxLocal2",
  "totalPrec_total",
  #"slope"
)

if(TRUE){ # model 
  if(TRUE){ # functions
    ensemble.model <- function(train_data0, test_data0, y_col0, x_cols0, use.models){ 
      library(caret) # needs to be inside function to export to cores
      if(FALSE){ # initialise vars for dev
        train_data0 <- feed.standardised
        test_data0 <- feed.standardised
        y_col0 <- y
        x_cols0 <- allXs
      } # initialise vars for dev
      formula <- as.formula(paste(y_col0, "~", paste(x_cols0, collapse=" + ")))  
      train_data0 <- train_data0[,c(y_col0, x_cols0)]
      test_data0 <- test_data0[,c(y_col0, x_cols0)]
      
      # train models to feed into linear ensemble
      models.train <- data.frame(y=train_data0[,y_col0])
      models.test <- data.frame(y=test_data0[,y_col0])
      fitControl <- trainControl(method="cv", number=5, savePredictions='final')
      
      if('rf' %in% use.models){ # random forest
        set.seed(1) 
        
        model_rf <- train(formula, data=train_data0, method='rf', trControl=fitControl, tuneLength=3)
        pred_rf_train <- predict(object=model_rf, train_data0)
        pred_rf_test <- predict(object=model_rf, test_data0)
        
        models.train <- cbind(models.train, rf=pred_rf_train)
        models.test <- cbind(models.test, rf=pred_rf_test)
      }        # random forest (takes forever to train)
      if('gbm' %in% use.models){ # gradient boosting
        set.seed(1)
        model_gbm <- train(formula,data=train_data0,method='gbm',trControl=fitControl,tuneLength=3,verbos=FALSE)
        pred_gbm_train <- predict(object = model_gbm,train_data0)
        pred_gbm_test <- predict(object = model_gbm,test_data0)
        
        models.train <- cbind(models.train, gbm=pred_gbm_train)
        models.test <- cbind(models.test, gbm=pred_gbm_test)
      }       # gradient boosting
      if('svm' %in% use.models){ # svms
        library(e1071) 
        model_svm <- svm(formula, data=train_data0)
        pred_svm_train <- predict(model_svm,train_data0,na.action=na.omit)
        pred_svm_test <- predict(model_svm,test_data0,na.action=na.omit)
        
        models.train <- cbind(models.train, svm=pred_svm_train)
        models.test <- cbind(models.test, svm=pred_svm_test)
      }       # svms
      if('knn' %in% use.models){ # knn 
        model_knn <- train(formula,data=train_data0,method='knn',trControl=fitControl,tuneLength=3)
        pred_knn_train <- predict(object = model_knn,train_data0)
        pred_knn_test <- predict(object = model_knn,test_data0)  
        
        models.train <- cbind(models.train, knn=pred_knn_train)
        models.test <- cbind(models.test, knn=pred_knn_test)
      }       # knn 
      if('xgb' %in% use.models){# xgboost (always use)
        library(xgboost) 
        
        model_xgboost <- xgboost(data = data.matrix(train_data0[,x_cols0]), 
                                 label = train_data0[,y_col0], 
                                 objective = "reg:squarederror",
                                 nrounds=15,
                                 verbose = FALSE)
        pred_xgboost_train <- predict(model_xgboost, data.matrix(train_data0[,x_cols0]))
        pred_xgboost_test <- predict(model_xgboost, data.matrix(test_data0[,x_cols0]))
        
        models.train <- cbind(models.train, xgb=pred_xgboost_train)
        models.test <- cbind(models.test, xgb=pred_xgboost_test)
        }       # xgboost (always use)
      if('rpart' %in% use.models){ # rpart (always use)
        library(rpart) #for rpart
        
        model_rpart <- rpart(formula, method="anova", data=train_data0,
                             control=rpart.control(minsplit=1, minbucket=1, cp=0.01, maxdepth=20))
        pred_rpart_train <- predict(model_rpart, newdata = train_data0)
        pred_rpart_test <- predict(model_rpart, newdata = test_data0)
        
        models.train <- cbind(models.train, rpart=pred_rpart_train)
        models.test <- cbind(models.test, rpart=pred_rpart_test)
        }     # rpart (not very useful)

      ensemble.model <- lm(y ~ ., data=models.train)
      ensemble.pred <- predict(ensemble.model, models.test, interval="predict")
      return(ensemble.pred)
    }
    combineFiles <- function(fileList0,d=1){
      #fileList0 <- paste0(wd_para,fl[((col-1)*factorial(nXs-1)+1):((col-1)*factorial(nXs-1)+factorial(nXs-1))])
      for (f in 1:1) {
        data<-read.csv(fileList0[f], stringsAsFactors = FALSE)
        data_all<-data
      }
      if(d==1){
        for (f in 2:length(fileList0)) {
          data<-read.csv(fileList0[f],stringsAsFactors = FALSE)
          data_all<-rbind(data_all,data)
        }
      }
      if(d==2){
        for (f in 2:length(fileList0)) {
          data<-read.csv(fileList0[f],stringsAsFactors = FALSE)
          data_all<-cbind(data_all,data)
        }
      }
      for (f in 1:length(fileList0)){
        fn <- paste0(fileList0[f])
        file.remove(fn)  
      }
      return(data_all)
    }# end of combineFiles
  } # functions
  if(TRUE){ # input
    data <- read.csv(infile, stringsAsFactors=FALSE)
    no_cores <- detectCores() - 1 
    wd_para <- paste0(wd,"/out_of_bag/")
    
    #format input file 
    if(TRUE){ 
      data$sd <- data$side
      data$tD <- data$trackDist
      data$cD <- data$coastDist
      idcols <- c('landing','sd','tD','cD')
    }
    
    #set up variables
    if(TRUE){
      # shift positive
      for(var in c(allXs)){
        if(min(data[,var], na.rm=TRUE) <= 0){
          data[,var] <- data[,var] + abs(min(data[,var], na.rm=TRUE)) + 1 
        }  
      }
    } 
  } # input
  
  if(TRUE){  # modelling
    output_all <- data.frame(matrix(nrow=0, ncol=27))
    colnames(output_all) <-  c( 'scale',
                                "y",
                                idcols,
                                "y.standardised",
                                'ensemble.pred','ensemble.pred.upper','ensemble.pred.lower',
                                'ensemble.pred.oob','ensemble.pred.oob.upper','ensemble.pred.oob.lower',
                                'ensemble.pred.oob.event','ensemble.pred.oob.event.upper','ensemble.pred.oob.event.lower',
                                'obs','ensemble.pred.rescale','ensemble.pred.upper.rescale','ensemble.pred.lower.rescale',
                                'ensemble.pred.oob.rescale','ensemble.pred.oob.upper.rescale','ensemble.pred.oob.lower.rescale',
                                'ensemble.pred.oob.event.rescale','ensemble.pred.oob.event.upper.rescale','ensemble.pred.oob.event.lower.rescale')
    start_time <- Sys.time()
    
    # loop through different scalings
    for(scl in c('log10')){ 
      # for debugging
      if(FALSE){ 
        scl <- 'log10' 
      }
      print(scl)  
      
      # loop y
      for (y in allYs){
        # for debugging
        if(FALSE){
          y <- allYs[1]
        } 
        
        print(y)
        feed <- data[, c(idcols, y, allXs)]
        feed.scaled <- feed
        
        if(TRUE){ # transform input
          if(scl=='log10'){
            for(x in c(y, allXs[!allXs %in% c('side')])){
              feed.scaled[,x] <- log10(feed.scaled[,x])
              feed.scaled[,x] [which(!is.finite(feed.scaled[,x] ))]<- NA 
            }
          }
          
          feed.scaled[,y] [which(!is.finite(feed.scaled[,y] ))] <- NA 
          feed.scaled <- feed.scaled[complete.cases(feed.scaled), ] 
        } # transform input
        
        feed.standardised <- feed.scaled
        for (col in c(y, allXs)){ #standardize input
          #col <- 3
          feed.standardised[,col] <- feed.standardised[,col] - mean(feed.standardised[,col])   
          feed.standardised[,col] <- feed.standardised[,col] / sd(feed.standardised[,col])   
        } #standardize input
        
        ensemble.pred <- as.data.frame(ensemble.model(feed.standardised, feed.standardised, y, allXs, use.models))
        
        if(FALSE){ # look at fit
          library(ModelMetrics)
          plot(feed.standardised[,y], ensemble.pred[,'fit'])
          rmse(feed.standardised[,y], ensemble.pred[,'fit'])
          mse(feed.standardised[,y], ensemble.pred[,'fit'])
          cov(feed.standardised[,y], ensemble.pred[,'fit']) ^ 2 # r-squared
        } # look at fit
        
        output <- data.frame(
          landing=feed.standardised$landing,
          sd=feed.standardised$sd,
          cD=feed.standardised$cD,
          tD=feed.standardised$tD,
          y.standardised=feed.standardised[,y],
          ensemble.pred=ensemble.pred$fit,
          ensemble.pred.upper=ensemble.pred$upr,
          ensemble.pred.lower=ensemble.pred$lwr
          )
        
        output$ensemble.pred.rescale <- 10^(output$ensemble.pred * sd(feed.scaled[,y]) + mean(feed.scaled[,y]))
        output$ensemble.pred.upper.rescale <- 10^(output$ensemble.pred.upper*sd(feed.scaled[,y]) + mean(feed.scaled[,y]))
        output$ensemble.pred.lower.rescale <- 10^(output$ensemble.pred.lower*sd(feed.scaled[,y]) + mean(feed.scaled[,y]))
        
        output <- merge(output,feed[,c(idcols,y)],by=c(idcols))
        colnames(output)[ncol(output)] <- 'obs'
        
        output$y <- y
        output$scale <- scl
        
        if(TRUE){ # out-of-bag
          cl <- makeCluster(no_cores)  
          clusterExport(cl, c("wd_para","feed.standardised",'idcols','y','allXs','ensemble.model', 'use.models'))
          
          parLapply(cl, 1:nrow(feed.standardised),
                    function(ind){
                      #ind<-2
                      train_data <- feed.standardised[-ind,]
                      test_data <- feed.standardised[ind,]
                      
                      pred.oob<- ensemble.model(train_data, test_data, y, allXs, use.models)
                      
                      output <- cbind(test_data[,idcols], pred.oob)
                      write.csv(output,file=paste0(wd_para, "predOOB", ind, ".csv"), row.names=FALSE)
                    })
          stopCluster(cl)
          
          fileList <- list.files(path=paste0(wd_para), pattern=paste0("predOOB"))
          ensemble.pred.oob <- combineFiles(paste0(wd_para,fileList), d=1)
          colnames(ensemble.pred.oob) <- c(idcols,"ensemble.pred.oob","ensemble.pred.oob.lower","ensemble.pred.oob.upper" )
          
          output <- merge(output, ensemble.pred.oob, by=c(idcols))
          
          output$ensemble.pred.oob.rescale <- 10^(output$ensemble.pred.oob*sd(feed.scaled[,y])+mean(feed.scaled[,y]))
          output$ensemble.pred.oob.upper.rescale <- 10^(output$ensemble.pred.oob.upper*sd(feed.scaled[,y])+mean(feed.scaled[,y]))
          output$ensemble.pred.oob.lower.rescale <- 10^(output$ensemble.pred.oob.lower*sd(feed.scaled[,y])+mean(feed.scaled[,y]))
          
        } # out-of-bag
        if(TRUE){ # eventwise out-of-bag
          eventList<-unique(feed.standardised$landing)
          cl <- makeCluster(no_cores)     
          clusterExport(cl,c('eventList',"wd_para","feed.standardised",'idcols','y','allXs','ensemble.model','use.models'))
          
          parLapply(cl,1:length(eventList),
                    function(id){
                      #id <-2
                      ind <- which(feed.standardised$landing==eventList[id])
                      train_data <- feed.standardised[-ind,]
                      test_data <- feed.standardised[ind,]
                      
                      pred.oob.event<- ensemble.model(train_data,test_data,y,allXs,use.models)
                      
                      output<-cbind(test_data[,idcols], pred.oob.event)
                      write.csv(output,paste0(wd_para,"predOOBEvent",id,".csv"),row.names=FALSE)
                    })
          stopCluster(cl)
          
          fileList <- list.files(path=paste0(wd_para), pattern=paste0("predOOBEvent"))
          ensemble.pred.oob.event<- combineFiles(paste0(wd_para,fileList), d=1)
          colnames(ensemble.pred.oob.event) <- c( idcols, "ensemble.pred.oob.event","ensemble.pred.oob.event.lower","ensemble.pred.oob.event.upper" )
          
          output <- merge(output, ensemble.pred.oob.event, by=c(idcols))
          
          output$ensemble.pred.oob.event.rescale <- 10^(output$ensemble.pred.oob.event*sd(feed.scaled[,y])+mean(feed.scaled[,y]))
          output$ensemble.pred.oob.event.upper.rescale <- 10^(output$ensemble.pred.oob.event.upper*sd(feed.scaled[,y])+mean(feed.scaled[,y]))
          output$ensemble.pred.oob.event.lower.rescale <- 10^(output$ensemble.pred.oob.event.lower*sd(feed.scaled[,y])+mean(feed.scaled[,y]))
        } # eventwise out-of-bag
        
        output_all <- rbind(output_all, output)
      } 
    } # loop through different scalings
    
    end_time <- Sys.time()
    print(end_time - start_time)
    
    output_all2 <- output_all[,c( 'scale',"y",idcols,"y.standardised",
                                  'ensemble.pred','ensemble.pred.upper','ensemble.pred.lower',
                                  'ensemble.pred.oob','ensemble.pred.oob.upper','ensemble.pred.oob.lower',
                                  'ensemble.pred.oob.event','ensemble.pred.oob.event.upper','ensemble.pred.oob.event.lower',
                                  'obs','ensemble.pred.rescale','ensemble.pred.upper.rescale','ensemble.pred.lower.rescale',
                                  'ensemble.pred.oob.rescale','ensemble.pred.oob.upper.rescale','ensemble.pred.oob.lower.rescale',
                                  'ensemble.pred.oob.event.rescale','ensemble.pred.oob.event.upper.rescale','ensemble.pred.oob.event.lower.rescale')]
    
    write.csv(output_all2, outfile, row.names=FALSE)
  } # modelling
} # model 
if(TRUE){ # plot results
  library(dplyr)
  library(tidyr)
  
  data <- read.csv(outfile, stringsAsFactors=FALSE)
  mdls <- c("ensemble.pred", "ensemble.pred.oob", "ensemble.pred.oob.event",
            "ensemble.pred.rescale", "ensemble.pred.oob.rescale", "ensemble.pred.oob.event.rescale")
  obList <- c("y.standardised", "y.standardised", "y.standardised",
              "obs", "obs", "obs")
  list <- unique(data[,c('scale','y')])
  pdf(file=outfig, width=8.5, height=11) 
  par(oma=c(3,1,3,1), mar=c(5,4,4,1), xpd=FALSE) 
  layout(matrix(seq(1, 12, 1), 4, 3, byrow=TRUE))
  
  for (row in 1:nrow(list)){
    #row <- 1 
    for (mm in 1:length(mdls)){
      mdl <- mdls[mm]
      ob <- obList[mm]
      
      # subset by scale and response var (e.g., log10 and intensity)
      data_sub <- data[which(data$scale==list[row,'scale'] & data$y==list[row,'y']),]
      
      r2 <- sum((data_sub[,mdl] - mean(data_sub[,ob]))^2) / sum((data_sub[,ob] - mean(data_sub[,ob]))^2)
      r2res <- 1 - sum((data_sub[,mdl] - data_sub[,ob])^2) / sum((data_sub[,ob] - mean(data_sub[,ob]))^2)
      mse <- mean((data_sub[,ob] - data_sub[,mdl])^2)
      mape <- mean(abs((data_sub[,ob] - data_sub[,mdl])) / data_sub[,ob])
      
      
      
      list[row,'n'] <- nrow(data_sub)
      list[row,paste0('r2_',mdl)] <- r2
      list[row,paste0('r2res_',mdl)] <- r2res
      list[row,paste0('mse_',mdl)] <- mse
      list[row,paste0('mape_',mdl)] <- mape
      
      plot(data_sub[,ob],data_sub[,mdl],
           xlab='obs',ylab=mdl,
           main=paste0(list[row,'y'],', ',
                       '\nen, ',list[row,'scale'],', ','n=',nrow(data_sub),',',
                       '\nr2=',round(r2, digits = 2) ,', r2res=',round(r2res,digits = 2),
                       '\nMSE=',round(mse, digits = 2) ,', MAPE=',round(mape,digits = 2))) 
      abline(a=0,b=1)
      #list[row,paste0('mse_',mdl)] <- sum((data_sub[,mdl] - data_sub[,'obs'])^2/(data_sub[,'obs'])^2)/nrow(data_sub)
      #list[row,paste0('mape_',mdl)] <- 1-sum(abs((data_sub[,mdl] - data_sub[,'obs'])/data_sub[,'obs']))/nrow(data_sub)
    }#loop row
  }#loop mdl
  
  dev.off()
  
  #write.csv(list,paste0(wd,'/result/model/ml/r2_en_3Method_rf_xgboost_rpart-storm6var.csv'),row.names = FALSE)
} # plot results
