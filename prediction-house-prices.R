#Loading data for predicting house prices in Bengaluru - train data
prices_train = read.csv(file.choose())

#checking the dimension of train data
dim(prices_train)

#looking top 5 rows
head(prices_train,5)

#Predictors/features matrix
x = prices_train[,c(1:8)]

#Response Vector
y = prices_train[,9]


#Data Cleaning
summary(prices_train)

unique(x$area_type)
#There are four different area types and can be converted to numerical representation for calculation purpose
#Built-up Area = 1
#Carpet Area = 2
#Plot Area = 3
#Super built-up Area = 4
x$area_type = as.numeric(x$area_type)


unique(x$size)
#As size shows the number of bedrooms - as BHK and Bedroom both shows the number of rooms, so created a column as # of bedroom
x$bedroom = 0
for (i in c(1:13320)){
x$bedroom[i] = as.numeric(strsplit(as.character(x$size)," ")[[i]][1])
}

#removing size column as it is no longer needed as it is replaced by bedroom column
x = x[,-4]

#As number of NAs are less so it can be replaced by mean value of bedroom
mu_bedroom = mean(x$bedroom, na.rm = TRUE)
x$bedroom[is.na(x$bedroom)] = mu_bedroom


unique(x$bath)
#As number of NAs are less so it can be replaced by mean value of bath
mu_bath = mean(x$bath, na.rm = TRUE)
x$bath[is.na(x$bath)] = mu_bath


unique(x$balcony)
#As number of NAs are less so it can be replaced by mean value of balcony
mu_balcony = mean(x$balcony, na.rm = TRUE)
x$balcony[is.na(x$balcony)] = mu_balcony


unique(x$total_sqft)
#As some of the rows in the total_sqft shows the range, so taking the mean of them for calculation purpose.
x$total_sqfts = 0
for (i in c(1:13320)){
  x$total_sqfts[i] = mean(as.numeric(strsplit(as.character(x$total_sqft),"-")[[i]]))
}

#removing total_sqft column as it is no longer needed as it is replaced by total_sqfts column
x = x[,-5]

#As number of NAs are less so it can be replaced by mean value of total_sqfts
mu_total_sqfts = mean(x$total_sqfts, na.rm = TRUE)
x$total_sqfts[is.na(x$total_sqfts)] = mu_total_sqfts



#Feature Scaling
#Calculating mean and range for each feature to normalize them
mu_area_type = mean(x$area_type)
ran_area_type = max(x$area_type)-min(x$area_type)

mu_bath = mean(x$bath)
ran_bath = max(x$bath)-min(x$bath)

mu_balcony = mean(x$balcony)
ran_balcony = max(x$balcony)-min(x$balcony)

mu_bedroom = mean(x$bedroom)
ran_bedroom = max(x$bedroom)-min(x$bedroom)

mu_total_sqfts = mean(x$total_sqfts)
ran_total_sqfts = max(x$total_sqfts)-min(x$total_sqfts)

#Normalizing the features
area_type_nor = (x$area_type - mu_area_type)/ran_area_type
bath_nor = (x$bath - mu_bath)/ran_bath
balcony_nor = (x$balcony - mu_balcony)/ran_balcony
bedroom_nor = (x$bedroom - mu_bedroom)/ran_bedroom
total_sqfts_nor = (x$total_sqfts - mu_total_sqfts)/ran_total_sqfts

x_zero = 1

x_norm = data.frame(x_zero,area_type_nor,bath_nor,balcony_nor,bedroom_nor,total_sqfts_nor)

head(x_norm,5)

#finding feature coefficients for linear regression i.e. theta
#Initializing theta with zero values
theta = matrix(0,6,1)

#Converting y into matrix
y = as.matrix(y)
x_norm = as.matrix(x_norm)

#USing gradient Descent to minize the cost function J.
alpha = 0.01
num_iter = 400
j_vals = matrix(0,1,401)
m = length(y)

j_vals[1] = (sum((x_norm%*%theta - y)^2))/(2*m)

for (i in c(2:401)){
  delta = (colSums((as.vector(x_norm%*%theta - y))*x_norm))/m
  delta_transpose = as.matrix(delta)
  theta = theta - (alpha*delta_transpose)
  
  j_vals[i] = (sum((x_norm%*%theta - y)^2))/(2*m)
}

j_vals = as.vector(j_vals)
plot(j_vals,xlab = 'num_iter',ylab = 'Value of cost function', main = 'Convergence Graph for Gradient descent')

#####################################################################################################
#Predicting the prices for test data:
#Loading test data to predict the prices
prices_test = read.csv(file.choose())

#checking the dimension of test data
dim(prices_test)

#looking top 5 rows
head(prices_test,5)

#Data Cleaning
summary(prices_test)

#Predictors/features matrix
x_test = prices_test[,c(1:8)]

x_test$area_type = as.numeric(x_test$area_type)

x_test$bedroom = 0
for (i in c(1:1480)){
x_test$bedroom[i] = as.numeric(strsplit(as.character(x_test$size)," ")[[i]][1])
}

#removing size column as it is no longer needed as it is replaced by bedroom column
x_test = x_test[,-4]

mu_bedroom_test = mean(x_test$bedroom, na.rm = TRUE)
x_test$bedroom[is.na(x_test$bedroom)] = mu_bedroom_test


mu_bath_test = mean(x_test$bath, na.rm = TRUE)
x_test$bath[is.na(x_test$bath)] = mu_bath_test


mu_balcony_test = mean(x_test$balcony, na.rm = TRUE)
x_test$balcony[is.na(x_test$balcony)] = mu_balcony_test

x_test$total_sqfts = 0
for (i in c(1:1480)){
x_test$total_sqfts[i] = mean(as.numeric(strsplit(as.character(x_test$total_sqft),"-")[[i]]))
}

x_test = x_test[,-5]

mu_total_sqfts_test = mean(x_test$total_sqfts, na.rm = TRUE)
x_test$total_sqfts[is.na(x_test$total_sqfts)] = mu_total_sqfts_test

#Feature Scaling for test data
#Normalizing the features
area_type_nor_test = (x_test$area_type - mu_area_type)/ran_area_type
bath_nor_test = (x_test$bath - mu_bath)/ran_bath
balcony_nor_test = (x_test$balcony - mu_balcony)/ran_balcony
bedroom_nor_test = (x_test$bedroom - mu_bedroom)/ran_bedroom
total_sqfts_nor_test = (x_test$total_sqfts - mu_total_sqfts)/ran_total_sqfts
x_zero_test = 1
x_norm_test = data.frame(x_zero_test,area_type_nor_test,bath_nor_test,balcony_nor_test,bedroom_nor_test,total_sqfts_nor_test)

predicted_price = as.matrix(x_norm_test)%*%theta
predicted_price = as.vector(predicted_price)

write.csv(predicted_price,"./house_price.csv",row.names = FALSE)
