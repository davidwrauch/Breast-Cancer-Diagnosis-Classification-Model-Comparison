# Breast Cancer Diagnosis Prediction
# Uses Wisconsin Diagnostic Breast Cancer features only
# Outcome: diagnosis, with levels B = benign and M = malignant

#-------------------------
# 0. Packages
#-------------------------

library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)
library(PRROC)
library(keras3)
library(tensorflow)

#-------------------------
# 1. Load and prepare data
#-------------------------

setwd("C:/data exercises/classification 2")

breast_cancer <- read.csv("breast-cancer.csv")

# Remove ID column if it exists
if ("id" %in% names(breast_cancer)) {
  breast_cancer$id <- NULL
}

# Make diagnosis a factor
# B = benign, M = malignant
breast_cancer$diagnosis <- factor(breast_cancer$diagnosis, levels = c("B", "M"))

str(breast_cancer)

# Predictor names: all columns except diagnosis
predictor_names <- setdiff(names(breast_cancer), "diagnosis")

#-------------------------
# 2. Train/test split
#-------------------------

set.seed(123)

trainIndex <- createDataPartition(
  breast_cancer$diagnosis,
  p = 0.7,
  list = FALSE
)

trainData <- breast_cancer[trainIndex, ]
testData  <- breast_cancer[-trainIndex, ]

#-------------------------
# 3. Logistic Regression
#-------------------------
# Scale predictors because logistic regression can benefit from standardized inputs.

pp_log <- preProcess(
  trainData[, predictor_names],
  method = c("center", "scale")
)

train_log_x <- predict(pp_log, trainData[, predictor_names])
test_log_x  <- predict(pp_log, testData[, predictor_names])

train_log <- train_log_x
train_log$diagnosis <- trainData$diagnosis

test_log <- test_log_x
test_log$diagnosis <- testData$diagnosis

log_formula <- as.formula(
  paste("diagnosis ~", paste(predictor_names, collapse = " + "))
)

log_model <- glm(
  log_formula,
  data = train_log,
  family = binomial
)

log_probs <- predict(log_model, newdata = test_log, type = "response")
log_preds <- ifelse(log_probs > 0.5, "M", "B")

log_cm <- confusionMatrix(
  factor(log_preds, levels = c("B", "M")),
  factor(test_log$diagnosis, levels = c("B", "M")),
  positive = "M"
)

print(log_cm)

#-------------------------
# 4. Random Forest
#-------------------------
# Tree models do not require scaling.

set.seed(123)

rf_model <- randomForest(
  diagnosis ~ .,
  data = trainData,
  ntree = 500,
  mtry = 5,
  importance = TRUE
)

rf_preds <- predict(rf_model, newdata = testData)
rf_probs <- predict(rf_model, newdata = testData, type = "prob")[, "M"]

rf_cm <- confusionMatrix(
  rf_preds,
  testData$diagnosis,
  positive = "M"
)

print(rf_cm)

# Optional: variable importance
varImpPlot(rf_model)

#-------------------------
# 5. XGBoost
#-------------------------
# XGBoost needs numeric matrix inputs and numeric 0/1 labels.

train_matrix <- model.matrix(diagnosis ~ ., data = trainData)[, -1]
test_matrix  <- model.matrix(diagnosis ~ ., data = testData)[, -1]

train_label <- ifelse(trainData$diagnosis == "M", 1, 0)
test_label  <- ifelse(testData$diagnosis == "M", 1, 0)

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)

# Cross-validation to choose number of rounds
set.seed(123)

cv_model <- xgb.cv(
  data = dtrain,
  nrounds = 200,
  nfold = 5,
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.1,
  early_stopping_rounds = 10,
  verbose = 0
)

best_nrounds <- cv_model$best_iteration
print(best_nrounds)

xgb_model <- xgb.train(
  data = dtrain,
  nrounds = best_nrounds,
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.1,
  verbose = 0
)

xgb_probs <- predict(xgb_model, newdata = dtest)

# Default threshold = 0.5
xgb_preds <- ifelse(xgb_probs > 0.5, 1, 0)

xgb_cm <- confusionMatrix(
  factor(xgb_preds, levels = c(0, 1)),
  factor(test_label, levels = c(0, 1)),
  positive = "1"
)

print(xgb_cm)

# Optional threshold check
thresholds <- seq(0.1, 0.9, by = 0.01)

threshold_results <- data.frame(
  threshold = thresholds,
  accuracy = sapply(thresholds, function(t) {
    mean(ifelse(xgb_probs > t, 1, 0) == test_label)
  })
)

plot(
  threshold_results$threshold,
  threshold_results$accuracy,
  type = "l",
  main = "XGBoost Accuracy vs. Threshold",
  xlab = "Threshold",
  ylab = "Accuracy"
)

#-------------------------
# 6. K-Nearest Neighbors
#-------------------------
# KNN is distance-based, so scaling is important.

set.seed(123)

knn_train_x <- trainData[, predictor_names]
knn_test_x  <- testData[, predictor_names]

knn_train_y <- trainData$diagnosis
knn_test_y  <- testData$diagnosis

ctrl <- trainControl(
  method = "cv",
  number = 5
)

knn_model <- train(
  x = knn_train_x,
  y = knn_train_y,
  method = "knn",
  trControl = ctrl,
  tuneLength = 10,
  preProcess = c("center", "scale")
)

print(knn_model)
plot(knn_model)

knn_preds <- predict(knn_model, newdata = knn_test_x)

knn_cm <- confusionMatrix(
  knn_preds,
  knn_test_y,
  positive = "M"
)

print(knn_cm)

#-------------------------
# 7. Keras Neural Network
#-------------------------
# Neural nets need scaled numeric predictors and numeric labels.

set.seed(123)

pp_nn <- preProcess(
  trainData[, predictor_names],
  method = c("center", "scale")
)

train_nn_x <- predict(pp_nn, trainData[, predictor_names])
test_nn_x  <- predict(pp_nn, testData[, predictor_names])

x_train <- as.matrix(train_nn_x)
x_test  <- as.matrix(test_nn_x)

y_train <- ifelse(trainData$diagnosis == "M", 1, 0)
y_test  <- ifelse(testData$diagnosis == "M", 1, 0)

input_dim <- ncol(x_train)

model <- keras_model_sequential() |>
  layer_dense(units = 16, activation = "relu", input_shape = input_dim) |>
  layer_dense(units = 8, activation = "relu") |>
  layer_dense(units = 1, activation = "sigmoid")

model |> compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history <- model |> fit(
  x_train,
  y_train,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2,
  verbose = 1
)

nn_probs <- model |> predict(x_test)
nn_preds <- ifelse(nn_probs > 0.5, 1, 0)

nn_cm <- confusionMatrix(
  factor(nn_preds, levels = c(0, 1)),
  factor(y_test, levels = c(0, 1)),
  positive = "1"
)

print(nn_cm)

#-------------------------
# 8. Compare Model Results
#-------------------------

model_results <- data.frame(
  model = c(
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "KNN",
    "Keras Neural Network"
  ),
  accuracy = c(
    log_cm$overall["Accuracy"],
    rf_cm$overall["Accuracy"],
    xgb_cm$overall["Accuracy"],
    knn_cm$overall["Accuracy"],
    nn_cm$overall["Accuracy"]
  ),
  sensitivity = c(
    log_cm$byClass["Sensitivity"],
    rf_cm$byClass["Sensitivity"],
    xgb_cm$byClass["Sensitivity"],
    knn_cm$byClass["Sensitivity"],
    nn_cm$byClass["Sensitivity"]
  ),
  specificity = c(
    log_cm$byClass["Specificity"],
    rf_cm$byClass["Specificity"],
    xgb_cm$byClass["Specificity"],
    knn_cm$byClass["Specificity"],
    nn_cm$byClass["Specificity"]
  )
)

print(model_results)

#-------------------------
# 9. Optional ROC/AUC
#-------------------------

log_roc <- roc(testData$diagnosis, log_probs, levels = c("B", "M"))
rf_roc  <- roc(testData$diagnosis, rf_probs, levels = c("B", "M"))
xgb_roc <- roc(testData$diagnosis, xgb_probs, levels = c("B", "M"))

print(auc(log_roc))
print(auc(rf_roc))
print(auc(xgb_roc))

plot(log_roc, main = "ROC Curves")
plot(rf_roc, add = TRUE)
plot(xgb_roc, add = TRUE)

legend(
  "bottomright",
  legend = c("Logistic Regression", "Random Forest", "XGBoost"),
  lwd = 2
)


rf_roc <- roc(testData$diagnosis, rf_probs, levels = c("B","M"))
auc(rf_roc)
#Area under the curve: 0.992

log_roc <- roc(testData$diagnosis, log_probs, levels = c("B","M"))
auc(log_roc)
#Area under the curve: 0.9633

varImpPlot(rf_model)
