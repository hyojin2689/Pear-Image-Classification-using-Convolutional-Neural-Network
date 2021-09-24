#Package Used 
library(tidyverse)
library(keras)

#Copying images to training and test directories
original_dataset_dir <- "C:/제이제이유/testpear"
base_dir <- "C:/pears"
dir.create(base_dir)

#사진을 넣을 빈 폴더 생성
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir,"test")
dir.create(test_dir)

train_pear_dir <- file.path(train_dir, "pear")
dir.create(train_pear_dir)

train_rottenpear_dir <- file.path(train_dir, "rottenpear")
dir.create(train_rottenpear_dir)

validation_pear_dir <- file.path(validation_dir,"pear")
dir.create(validation_pear_dir)

validation_rottenpear_dir <- file.path(validation_dir, "rottenpear")
dir.create(validation_rottenpear_dir)

test_pear_dir <- file.path(test_dir,"pear")
dir.create(test_pear_dir)

test_rottenpear_dir <- file.path(test_dir,"rottenpear")
dir.create(test_rottenpear_dir)

a <- sample(x=1:2862,size=2000,replace=F)
b <- sample(x=1:2866,size=2000,replace=F) 

fnames <- paste0("pear (", a[1:1000],").JPG")
file.copy(file.path(original_dataset_dir,fnames),
          file.path(train_pear_dir))

fnames <- paste0("pear (", a[1001:1500] ,").JPG")
file.copy(file.path(original_dataset_dir,fnames),
          file.path(validation_pear_dir))

fnames <- paste0("pear (", a[1501:2000] ,").JPG")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_pear_dir))

fnames <- paste0("rottenpear (", b[1:1000],").JPG")
file.copy(file.path(original_dataset_dir,fnames),
          file.path(train_rottenpear_dir))

fnames <- paste0("rottenpear (", b[1001:1500], ").JPG")
file.copy(file.path(original_dataset_dir,fnames),
          file.path(validation_rottenpear_dir))

fnames <- paste0("rottenpear (",b[1501:2000],").JPG")
file.copy(file.path(original_dataset_dir,fnames),
          file.path(test_rottenpear_dir))

cat("total training pear images", length(list.files(train_pear_dir)),"\n")
cat("total training rottenpear images", length(list.files(train_rottenpear_dir)),"\n")
cat("total validation pear images", length(list.files(validation_pear_dir)),"\n")
cat("total validation rottenpear images", length(list.files(validation_rottenpear_dir)),"\n")
cat("total test pear images",length(list.files(test_pear_dir)),"\n")
cat("total test rottenpear images",length(list.files(test_rottenpear_dir)),"\n")

#Building network
model <- keras_model_sequential() %>%
  layer_conv_2d(input_shape = c(150, 150, 3), filters = 16, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2))

#Adding a classifier to the convnet
model <- model %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 1, activation ='sigmoid')

#Compile: Configuring a Keras model for training
model%>%compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  metrics=c("acc")
)

#Data preprocessing
train_datagen <- image_data_generator(rescale=1/255)
validation_datagen <- image_data_generator(rescale=1/255)

train_generator <- flow_images_from_directory(
 train_dir,
  train_datagen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = "binary"
)

#Training the Neural Network
##Implementing a data generator for the test images
histroy <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs =10,
  validation_data = validation_generator,
  validation_steps = 50
)

model%>%save_model_hdf5("pears1.h5")

##Generating predictions on new data/our own data
fun_dir <- file.path(base_dir, "my_test_images")
dir.create(fun_dir)

pearrr_dir <- file.path(fun_dir, "my_images")
dir.create(pearrr_dir)

download_imgs <- list.files(path = "C:/pears/test/pear",
                            pattern = ".JPG",
                            full.names = T )

download_imgs2 <- list.files(path = "C:/pears/test/rottenpear",
                            pattern = ".JPG",
                            full.names = T )

download_imgs3 <- c(download_imgs,download_imgs2)

download_imgs4 <- sample(download_imgs3,1000,replace = F)

download_imgs5<-c()

for(i in 1:length(download_imgs3)){
  download_imgs5<-c(download_imgs5,unlist(str_split(download_imgs4[i],"/"))[5])
}

file.copy(file.path(test_pear_dir, download_imgs5),
          file.path(pearrr_dir))

file.copy(file.path(test_rottenpear_dir, download_imgs5),
          file.path(pearrr_dir))
 
#Shuffle images in file (외부 프로그램 사용, 이미지에 무작위성 부여) 
 
test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
  directory = fun_dir,
  generator = test_datagen,
  target_size = c(150, 150),
  batch_size = 1000,
  class_mode = 'binary',
  shuffle = F
)

#Generating predictions for the test samples from a data generator
predictions <- model %>% predict_generator(
  steps = 1,
  generator = test_generator,
  verbose = 0
)

image_labels <- list.files(path = pearrr_dir)

pred_results <- as.data.frame(cbind(image_labels, predictions)) %>%
  rename("prediction" = 2) %>%
  mutate("predicted_class" = if_else(prediction>0.5,print("rottenpear"),print("pear")),
         prediction = as.double(prediction))
