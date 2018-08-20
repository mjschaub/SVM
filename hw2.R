library(caret)
setwd('D:/CS498/HW2 - SVM/')

raw_data_train <- read.csv('adult_data.txt', header=FALSE, na.strings = "?")
raw_data_test <- read.csv('adult_test.txt', header=FALSE, na.strings = "?")
raw_data <- rbind(raw_data_train, raw_data_test, make.row.names=FALSE)

x_data <- raw_data[,c(1,3,5,11,12,13)]
labels <- raw_data[,15]

#x_data[1:6]
for (i in 1:6)
{
  x_data[i] <- scale(as.numeric(as.matrix(x_data[i])))
}

data_partition <- createDataPartition(y=labels, p=.8, list=FALSE) #.8 for training
x_train <- x_data[data_partition,]
y_train <- labels[data_partition]
x_temp <- x_data[-data_partition,]
y_temp <- labels[-data_partition]
data_partition_two <- createDataPartition(y=y_temp, p=.5, list=FALSE) #.1 for each testing and validation
x_test <- x_temp[data_partition_two,]
y_test <- y_temp[data_partition_two]
x_validation <- x_temp[-data_partition_two,]
y_validation <- y_temp[-data_partition_two]


reg_constants <- c(.001,.005,.0075,.01, .1, 1)
num_epochs <- 50
steps <- 300
plot_steps <- 30
step_size_one <- .03
step_size_two <- 40 

calculate_ax_equals_b <- function(x, A, b)
{
  next_x <- as.numeric(as.matrix(x))
  return (t(A) %*% next_x + b)  #Ax+b
}

less_than_one <- labels[48820] #<=50K.
less_than_two <- labels[30000] #<=50K
greater_than_one <- labels[48210] #>50K.
greater_than_two <- labels[8] #>50K


check_greater_less_than <- function(y_label)
{
  if(y_label == greater_than_one | y_label == greater_than_two)
  {
    return (1)
  }
  else if(y_label == less_than_one | y_label == less_than_two)
  {
    return (-1)
  }
  else
  {
    return(NA)
  }
}

accuracy <- function(x,y,a,b)
{
  correct <- 0
  wrong <- 0
  for (i in 1:length(y))
  {
    prediction <- calculate_ax_equals_b(x[i,], a, b)
    if(prediction >= 0)
      prediction <- 1
    else
      prediction <- -1
    
    correct_val <- check_greater_less_than(y[i])
    
    if(prediction == correct_val)
      correct <- correct + 1 
    else
      wrong <- wrong + 1
    
  }
  return (c(correct/(correct+wrong)))
}

test_accuracies = c()
validate_accuracies = c()

for (reg in reg_constants)
{
  b <- 0
  a <- c(0,0,0,0,0,0)
  
  accuracies <- c()
  magnitude_coeff_vec <- c()
  num_pos <- 0
  num_neg <- 0
  for(epoch in 1:num_epochs)
  {
    
    #get 50 examples for evaluating
    random_vals <- sample(1:dim(x_train)[1], 50)
    accuracy_data <- x_train[random_vals,]
    accuracy_labels <- y_train[random_vals]
    train_data <- x_train[-random_vals,]
    train_labels <- y_train[-random_vals]
    
    curr_steps <- 0
    for(s in 1:steps)
    {
      
      if(curr_steps %% plot_steps == 0)
      {
        calc_accuracy <- accuracy(accuracy_data, accuracy_labels, a, b)
        accuracies <- c(accuracies, calc_accuracy)
        norm_a = norm(a,type="2")
        magnitude_coeff_vec <- c(magnitude_coeff_vec,norm_a)
      }
      
      samples <- sample(1:length(train_labels),1)
      while(is.na(check_greater_less_than(train_labels[samples])))
      {
        samples <- sample(1:length(train_labels),1)
      }
      x_examples <- as.numeric(as.matrix(train_data[samples,]))
      y_examples <- check_greater_less_than(train_labels[samples])
      
      prediction <- calculate_ax_equals_b(x_examples, a, b)
      step_size = 1 / ((step_size_one * epoch) + step_size_two)
      
      
      if(y_examples*prediction < 1)
      {
        num_neg <- num_neg + 1
        grad_vec_a <- (reg * a) - (y_examples*x_examples)
        grad_vec_b <- -(y_examples)
      }
      else
      {
        num_pos <- num_pos + 1
        grad_vec_a <- reg * a
        grad_vec_b <- 0
      } 
      
      #calc new a and b
      a <- a - (step_size * grad_vec_a)
      b <- b - (step_size * grad_vec_b)
      curr_steps <- curr_steps + 1
    }
  }
  
  new_validate_acc <- accuracy(x_validation, y_validation, a, b)
  validate_accuracies <- c(validate_accuracies, new_validate_acc)
  new_test_acc <- accuracy(x_test, y_test, a, b)
  test_accuracies <- c(test_accuracies, new_test_acc)
  
  #plot accuracies every 30 steps for a total of 500 points
  accuracy_string = paste("accuracies",toString(reg),sep="_")
  jpeg(file=paste(accuracy_string,".jpg"))
  header<- paste("reg_const = ",toString(reg),"Accuracies")
  plot(1:length(accuracies), accuracies, type="o", col="red", xlab ="30 step intervals", ylab ="Accuracy", main=header)
  dev.off()
  #plot magnitude of coefficient vector every 30 steps for a total of 500 points
  magnitude_string = paste("magnitude",toString(reg),sep="_")
  jpeg(file=paste(magnitude_string,".jpg"))
  header<- paste("coefficient vector = ",toString(reg),"Magnitudes")
  plot(1:length(magnitude_coeff_vec), magnitude_coeff_vec, type="o", col="red", xlab ="30 step intervals", ylab="Magnitude", main =header)
  dev.off()
  
}

max_idx <- 1
for(i in 1:length(validate_accuracies))
{
  if(validate_accuracies[i] >= validate_accuracies[max_idx])
    max_idx <- i
}
max_reg_constant <- reg_constants[max_idx]
max_reg_constant


test_accuracies[max_idx]





