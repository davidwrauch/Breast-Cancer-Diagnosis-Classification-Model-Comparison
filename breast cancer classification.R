#data from here https://www.kaggle.com/datasets/zahranusrat/social-media-advertising-response-data/data


install.packages("randomForest")
install.packages("randomForest")
install.packages("caret")
install.packages("lubridate")
install.packages("S7")
install.packages("ggplot2")
install.packages("forecast")
install.packages("rpart.plot")
install.packages("precrec")
install.packages("PRROC")
install.packages("xgboost")
install.packages("neuralnet")
install.packages("keras3")
install.packages("reticulate")
install.packages("tensorflow")
install_keras()

library(reticulate)
library(keras3)
library(neuralnet)
library('forecast')
library('tseries')
library(ggplot2)
library(randomForest)
library(caret)
library(readxl)
library(lubridate)
library(tidyverse)
library(data.table)
library(dplyr)
library(readr)
library(stringr)
library(stringi)
library(openxlsx)
library(rpart.plot)
library(rpart)
library(tseries)
library(pROC)
library(PRROC)
library(xgboost)
library(tensorflow)

# This is the new way to install TensorFlow
#install_tensorflow()


setwd("C:/data exercises/classification 2")

breast_cancer <- read.csv('breast-cancer.csv')
str(breast_cancer)

library(caret)

#-------------------------
# 1. Prepare the data
#-------------------------

df <- breast_cancer

# Remove ID column
df$id <- NULL

# Convert diagnosis to factor with 0/1
df$diagnosis <- factor(df$diagnosis, levels = c("B","M"))  # B = 0, M = 1

#-------------------------
# 2. Train/test split
#-------------------------
set.seed(123)
idx <- createDataPartition(df$diagnosis, p = 0.7, list = FALSE)

trainData <- df[idx, ]
testData  <- df[-idx, ]

#-------------------------
# 3. Scale all numeric predictors
#-------------------------

predictor_names <- setdiff(names(df), "diagnosis")

pp <- preProcess(trainData[, predictor_names], method = c("center", "scale"))

train_scaled <- predict(pp, trainData[, predictor_names])
test_scaled  <- predict(pp, testData[,  predictor_names])

# recombine with outcome
train_scaled$diagnosis <- trainData$diagnosis
test_scaled$diagnosis  <- testData$diagnosis

#-------------------------
# 4. Fit logistic regression
#-------------------------

log_formula <- as.formula(
  paste("diagnosis ~", paste(predictor_names, collapse = " + "))
)

log_model <- glm(
  log_formula,
  data = train_scaled,
  family = binomial
)

summary(log_model)   # optional

#-------------------------
# 5. Predict on test set
#-------------------------

prob <- predict(log_model, newdata = test_scaled, type = "response")
pred_class <- ifelse(prob > 0.5, "M", "B")

#-------------------------
# 6. Confusion Matrix
#-------------------------

confusionMatrix(
  factor(pred_class, levels = c("B","M")),
  factor(test_scaled$diagnosis, levels = c("B","M"))
)
#.95, sens .934, spec .984




########## now try random forest
breast_cancer$id <- NULL

# Convert diagnosis to factor (M = malignant, B = benign)
breast_cancer$diagnosis <- factor(breast_cancer$diagnosis, levels = c("B", "M"))

# Train/test split
set.seed(123)
trainIndex <- createDataPartition(breast_cancer$diagnosis, p = 0.7, list = FALSE)
trainData <- breast_cancer[trainIndex, ]
testData  <- breast_cancer[-trainIndex, ]

# Random forest model using all predictors
rf_model <- randomForest(
  diagnosis ~ .,              # use all remaining variables
  data = trainData,
  ntree = 500,                # number of trees
  mtry = 5,                   # number of variables tried at each split (tuneable)
  importance = TRUE
)

# Predict class labels
rf_preds <- predict(rf_model, newdata = testData)

# Predict probabilities (for ROC/PR curves)
rf_probs <- predict(rf_model, newdata = testData, type = "prob")[,2]

# Confusion matrix
confusionMatrix(rf_preds, testData$diagnosis)
#.99, sens .98, spec 1




####stopped here bc we're already getting amazing results, don't need anymore more complicated













####

##########now xgboost

# Training and test matrices
train_matrix <- model.matrix(diagnosis ~ Age + EstimatedSalary, data = trainData)[,-1]
test_matrix  <- model.matrix(diagnosis ~ Age + EstimatedSalary, data = testData)[,-1]

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)


# Labels must be numeric (0/1)
train_label <- as.numeric(trainData$diagnosis)
test_label  <- as.numeric(testData$diagnosis)


#run xgboost default
xgb_model <- xgb.train(
  data = dtrain,
  nrounds = 100,
  objective = "reg:logistic",   # classification objective
  max_depth = 3,
  eta = 0.1,
  verbose = 0
)

# Predict probabilities
xgb_probs <- predict(xgb_model, newdata = dtest)

# Convert to class labels
xgb_preds <- ifelse(xgb_probs > 0.311, 1, 0)

confusionMatrix(factor(xgb_preds), factor(test_label))
#.917 accuracy
#.81 kappa
#sensitivity .93
#specificity .90


#####trying xgboost again with using cross validation to find the ideal number of rounds
cv_model <- xgb.cv(
  data = dtrain,
  nrounds = 200,              # upper limit
  nfold = 5,                  # 5-fold CV
  objective = "reg:logistic", # classification objective
  max_depth = 3,
  eta = 0.1,
  metrics = "auc",            # optimize for AUC
  early_stopping_rounds = 10, # stop if no improvement in 10 rounds
  verbose = 0
)

# Look at the evaluation log
head(cv_model$evaluation_log)

# Best iteration is the one with the lowest test-error or highest test-auc
best_nrounds <- which.max(cv_model$evaluation_log$test_auc_mean)
best_nrounds

xgb_model_nrounds <- xgb.train(
  data = dtrain,
  nrounds = best_nrounds,
  objective = "reg:logistic",
  max_depth = 3,
  eta = 0.1,
  verbose = 0
)

# Predict probabilities
xgb_probs_nrounds <- predict(xgb_model_nrounds, newdata = dtest)

# Convert to class labels
xgb_preds_nrounds <- ifelse(xgb_probs_nrounds > 0.311, 1, 0)


confusionMatrix(factor(xgb_preds_nrounds), factor(test_label))
#.933 accuracy
#.85 kappa
#sensitivity .92
#specificity .95
#slightly better than untuned xgboost



####setting up a grid search so I can automatically test combinations of XGBoost hyperparameters and pick the best options. This way I don't have to manually tweak knobs like max_depth, eta, or subsample.
train_label <- as.numeric(trainData$diagnosis)
dtrain <- xgb.DMatrix(data = as.matrix(train_matrix), label = train_label)

param_grid <- expand.grid(
  max_depth = c(3, 5),
  eta = c(0.05, 0.1),
  subsample = c(0.7, 1.0),
  colsample_bytree = c(0.7, 1.0),
  min_child_weight = c(1, 5)
)

results <- list()

for (i in 1:nrow(param_grid)) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "error",  # <-- accuracy = 1 - error
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    min_child_weight = param_grid$min_child_weight[i]
  )
  
  set.seed(123)
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 200,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  best_iter <- cv$best_iteration
  best_error <- min(cv$evaluation_log$test_error_mean)
  best_acc <- 1 - best_error
  
  results[[i]] <- c(params, best_iter = best_iter, best_acc = best_acc)
}

results_df <- do.call(rbind.data.frame, results)
results_df[] <- lapply(results_df, function(x) as.numeric(as.character(x)))
results_df <- results_df[order(-results_df$best_acc), ]


best_params <- results_df[1, ]
params <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight
)

xgb_model_grid <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,   
  verbose = 0
)
xgb_probs_grid <- predict(xgb_model_grid, newdata = dtest)


preds <- predict(xgb_model_grid, newdata = as.matrix(test_matrix))
thresholds <- seq(0.3, 0.7, by = 0.01)
acc <- sapply(thresholds, function(t) mean(ifelse(preds > t, 1, 0) == test_label))
plot(thresholds, acc, type = "l", main = "Accuracy vs. Threshold")



# Convert to class labels
xgb_preds_grid <- ifelse(xgb_probs_grid > 0.3111, 1, 0)

confusionMatrix(factor(xgb_preds_grid), factor(test_label))
#.933 accuracy
#.85 kappa
#sensitivity .93
#specificity .95
#same as best_nrounds xgboost, oh well





##########trying with k-nearest neighbor

# Make sure your labels are factors with two levels
train_label <- factor(ifelse(trainData$diagnosis == 1, "Yes", "No"),
                      levels = c("No","Yes"))
test_label  <- factor(ifelse(testData$diagnosis == 1, "Yes", "No"),
                      levels = c("No","Yes"))

train_matrix <- data.frame(
  Age = as.numeric(trainData$Age),
  EstimatedSalary = as.numeric(trainData$EstimatedSalary)
)

test_matrix <- data.frame(
  Age = as.numeric(testData$Age),
  EstimatedSalary = as.numeric(testData$EstimatedSalary)
)

# Control setup
ctrl <- trainControl(method = "cv", number = 5)

# Train KNN
knn_model <- train(
  x = train_matrix,
  y = train_label,
  method = "knn",
  trControl = ctrl,
  tuneLength = 10
)

# Inspect results
print(knn_model)
plot(knn_model)

preds <- predict(knn_model, newdata = test_matrix)
confusionMatrix(preds, test_label)
#.83 accuracy
#sensitivty is .94
#specificity is .59, pretty bad

#so let's try again while scaling
knn_model <- train(
  x = train_matrix,
  y = train_label,
  method = "knn",
  trControl = ctrl,
  tuneLength = 10,
  preProcess = c("center","scale")   # normalize features
)

preds <- predict(knn_model, newdata = test_matrix)
confusionMatrix(preds, test_label)
#.91 accuracy
#sensitivity .93
#specificity .90






#try a neural net, r's first, neuralnet: A classical feedforward neural network implementation in R. Designed mainly for educational use and small-scale problems.

#Normalize predictors (important for neural networks)


breast_cancer_nn<- breast_cancer

#$convert to scale
breast_cancer_nn$Age <- scale(breast_cancer_nn$Age)
breast_cancer_nn$EstimatedSalary <- scale(breast_cancer_nn$EstimatedSalary)

# 70-30 train/test split
trainIndex <- createDataPartition(breast_cancer_nn$diagnosis, p = 0.7, list = FALSE)
trainData_nn <- breast_cancer_nn[trainIndex, ]
testData_nn  <- breast_cancer_nn[-trainIndex, ]

# Neural network with one hidden layer of 5 neurons
nn_model <- neuralnet(diagnosis ~ Age + EstimatedSalary,
                      data = trainData_nn,
                      hidden = 5,
                      linear.output = FALSE)

plot(nn_model)

# Compute predictions
nn_results <- compute(nn_model, testData_nn[, c("Age", "EstimatedSalary")])

# Extract probabilities
probabilities <- nn_results$net.result

# Convert to binary (threshold = 0.5)
predicted <- ifelse(probabilities > 0.5, 1, 0)

# Evaluate accuracy
confusionMatrix(factor(predicted), factor(testData_nn$diagnosis))
#.85 accuracy, not great

#For small, tabular datasets with a mix of numeric/categorical features, tree‑based ensembles (XGBoost, LightGBM, Random Forest) usually outperform neural nets.

#For large, high‑dimensional, unstructured data, neural nets (deep learning) are the go‑to.






###############trying keras neural network, A high-level API for deep learning, backed by TensorFlow. Supports complex, large-scale architectures (CNNs, RNNs, transformers, etc.).

idx <- createDataPartition(breast_cancer$diagnosis, p = 0.7, list = FALSE)
train <- breast_cancer[idx, ]
test  <- breast_cancer[-idx, ]

# --- 2. Scale predictors (important for neural nets) ---
pp <- preProcess(train[, c("Age","EstimatedSalary")], 
                 method = c("center","scale"))

train_scaled <- predict(pp, train[, c("Age","EstimatedSalary")])
test_scaled  <- predict(pp,  test[, c("Age","EstimatedSalary")])

# --- 3. Convert to matrices ---
x_train <- as.matrix(train_scaled)
y_train <- as.numeric(train$diagnosis)

x_test <- as.matrix(test_scaled)
y_test <- as.numeric(test$diagnosis)

# ✔ Correct: input_shape must be ncol(x_train), NOT ncol(x)
input_dim <- ncol(x_train)

# --- 4. Build the model (simple + no dropout = better for tabular data) ---
model <- keras_model_sequential() |>
  layer_dense(units = 4, activation = "relu", input_shape = input_dim) |>
  layer_dense(units = 1, activation = "sigmoid")

model |> compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.01),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

# --- 5. Train ---
history <- model |> fit(
  x_train, y_train,
  epochs = 150,
  batch_size = 16,
  validation_split = 0.2
)

# --- 6. Predict on test set ---
pred_prob <- model |> predict(x_test)
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# --- 7. Confusion Matrix ---
confusionMatrix(
  factor(pred_class, levels = c(0,1)),
  factor(y_test,    levels = c(0,1))
)
