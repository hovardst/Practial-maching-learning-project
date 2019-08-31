library(ggplot2)
library(caret)
library(AppliedPredictiveModeling)
library(kernlab)

#function to draw confusion matrix based on  https://stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package 
#and https://stackoverflow.com/questions/21589991/plot-a-confusion-matrix-with-color-and-frequency-in-r

confMatPlot = function(cm, titleMy, shouldPlot = T) {
  #' Function for plotting confusion matrice
  #' 
  #' @param cm: confusion matrix with counts, ie integers. 
  #' Fractions won't work
  #' @param titleMy: String containing plot title
  #' @return Nothing: It only plots
  
  ## Prepare data
  x.orig = cm$table  # Lazy conversion to function internal variable name
  n = nrow(x.orig)  # conf mat is square by definition, so nrow(x) == ncol(x)
  opar <- par(mar = c(5.1, 8, 3, 2))
  x <- x.orig
  x <- log(x + 0.5)  # x<1 -> x<0 ,  x>=1 -> x>0
  x[x < 0] <- NA
  diag(x) <- -diag(x)  # change sign to give diagonal different color
  
  ## Plot confusion matrix
  layout(matrix(c(1,1,2)))
  
  image(1:n, 1:n,  # grid of coloured boxes
        # matrix giving color values for the boxes
        # t() and [,ncol(x):1] since image puts [1,1] in bottom left by default
        -t(x)[, n:1],  
        # x and ylab added later to avoid overlap with tick labels
        xlab = '', ylab = '',
        col = colorRampPalette(c("darkorange3", "white", "steelblue"), 
                               bias = 1.65)(100),
        xaxt = 'n', yaxt = 'n'
  )
  # Plot counts
  text(rep(1:n, each = n), rep(n:1, times = n), 
       labels = sub('^0$', '', round(c(x.orig), 0)))
  # Axis ticks but no lables
  axis(1, at = 1:n, labels = rep("", n), cex.axis = 0.8)
  axis(2, at = n:1, labels = rep("", n), cex.axis = 0.8)
  # Tilted axis lables
  text(cex = 0.8, x = (1:n), y = -0.1, colnames(x), xpd = T,  adj = 1)
  text(cex = 0.8, y = (n:1), x = +0.1, colnames(x), xpd = T,  adj = 1)
  title(main = titleMy)
  title(ylab = 'Predicted', line = 6)
  title(xlab = 'Actual', line = 4)
  # Grid and box
  abline(h = 0:n + 0.5, col = 'gray')
  abline(v = 0:n + 0.5, col = 'gray')
  box(lwd = 1, col = 'gray')
  par(opar)
  
  
  
  # add in the specifics 
  nam <- colnames(cm$byClass) #extract names or columns in confusion matrix
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, nam[1], cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, nam[2], cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, nam[5], cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, nam[6], cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, nam[7], cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
  
  #save plot as PNG
  fname <- paste (titleMy,".png", sep = "")
  dev.copy(png,filename=fname, width=1144,height=860, res=120 )
  dev.off ()
}



# read CSV testdata from Web. Info : http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har
pml_training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))

#read test data from web
pml_test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))

#do manual cleaing of data with various methodes
#first clean data, removing variables wiht near zearo variance
zerovar <- nearZeroVar(pml_training, saveMetrics = TRUE)
pml_training_nzv <- pml_training [,zerovar$nzv == "FALSE"]

#removing first variabel X which is index,  user_name and cvtd_timestamp  that are not adding info for traning and prediction
pml_training_nzv <- pml_training_nzv [, -c(1,2,5)]

#some of the variables has a large propotion of NA values. Remove all variables that have more than 80% NA values. 
naclIndex <- colSums(is.na(pml_training_nzv))/nrow(pml_training_nzv) < 0.80
pml_training_clean <- pml_training_nzv[,naclIndex]



#Split trainingdata to get traing and crossvalidation set for manualy cleaned data set
inTrain <- createDataPartition(pml_training_clean$classe, p =0.8, list=FALSE)
training_cl <- pml_training_clean[ inTrain,]
crossval_cl <- pml_training_clean[-inTrain,]


#initate parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

#Use Linear Discriminant Analysis (LDA) metode on manualy cleaned training data 
trCtrl <- trainControl(method = "cv", number = 5)
LDAmodel  <- train(classe ~.,data = crossval_cl, method="lda",  trControl = trCtrl, metric = "Accuracy", allowParallel=TRUE)


#predict based om LDA modell for crossval set and compare predicted outcome with actual outcome
LDAmodell_pred = predict(LDAmodel, crossval_cl)
LDAcm <- confusionMatrix(LDAmodell_pred, crossval_cl$classe)
confMatPlot(LDAcm, "Confusion matrix - LDA modell")

#Use random forest metode on trainingdata 
RFmodel  <- train(classe ~.,data = training_cl, method="rf", tuneGrid = data.frame(mtry = 5))

#predict based om RF modell for crossval set and compare predicted outcome with actual outcome
RFmodell_pred = predict(RFmodel, crossval_cl)
RFcm <- confusionMatrix(RFmodell_pred, crossval_cl$classe)
confMatPlot(RFcm, "Confusion matrix - RF modell")

#de-registrer parallel cluster
stopCluster(cluster)
registerDoSEQ()