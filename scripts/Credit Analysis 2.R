#Things to install
install.packages("DescTools")
install.packages("smotefamily")
install.packages("car")
install.packages("tidyr")
install.packages("randomForest")
install.packages("xgboost")
library(xgboost)
library(randomForest)
library(smotefamily)
library(dplyr)
library(ggplot2) #plot distribution
library(e1071) #to check skewness
library(mice) #for predictive imputation
library(lubridate) #to convert dates to duration
library(DescTools) #to winsorize
library(caret) # to transform data
library(fastDummies) #categorical variable encoding
library(caTools) #to split dataset
library(car) #to check for collinearity
library(tidyr) #to troubleshoot
library(glmnet) #lasso regression
library(pROC) #AUC-ROC calculation
#EXPLORATORY DATA ANALYSIS
#Import dataset
dataset = read.csv('Credit Analysis Data.csv')
#Summary Statistics
str(dataset)
#There are 51 Numeric Varaiables and 23 Character variables
summary(dataset)
head(dataset)
colSums(is.na(dataset))
#Checking Balance of Target Variable 
table(dataset$loan_status)  # Count of each loan status
prop.table(table(dataset$loan_status))  
#Plotting a graph of the different classes           
ggplot(dataset, aes(x = loan_status)) +
  geom_bar(fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution of Loan Status", x = "Loan Status", y = "Count") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))
#Discard current/issued/na loan status
dataset <- dataset[!is.na(dataset$loan_status) & !dataset$loan_status %in% c("Issued", "Current"), ]
#Current Distribution 
ggplot(dataset, aes(x = loan_status)) +
  geom_bar(fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution of Loan Status", x = "Loan Status", y = "Count") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))
#Create New binary variable for loan_status 
dataset <- dataset %>%
  mutate(loan_status_binary = case_when(
    loan_status == "Fully Paid" ~ 1,   
    loan_status == "Charged Off" ~ 0, 
    loan_status == "Default" ~ 0,
    loan_status == "Late (31-120 days)" ~ 0,
    loan_status == "In Grace Period" ~ 0,
    loan_status == "Late (16-30 days)" ~ 0,
    loan_status == "Does not meet the credit policy. Status:Fully Paid" ~ 1,
    loan_status == "Does not meet the credit policy. Status:Charged Off" ~ 0,
    TRUE ~ as.numeric(loan_status))) 
#Discard irrelevant columns 
dataset <- dataset %>% select(-id,-member_id,-url,-desc,-title, -emp_title, -recoveries,-collection_recovery_fee,-verification_status_joint, -zip_code, -policy_code) 
#Handling Missingness in Data
#Checking percentage of missingness
colSums(is.na(dataset)) #sum of NAs in each dataset
na_percentage <- colSums(is.na(dataset)) / nrow(dataset) * 100 #percentage of NAs
na_percentage_2dp <- round(na_percentage,2) #percentage rounded to 2dp
print(na_percentage)
#Identifying skewness in select variables to decide whether to impute with mean or median
skewness_values <- apply(dataset, 2, function(x) skewness(as.numeric(x), na.rm = TRUE))
print(skewness_values)
#Median/Mean imputed as per skewness
replace_with_mean <- c("revol_util","total_acc")
replace_with_median <- c("delinq_2yrs","inq_last_6mths","open_acc","pub_rec","collections_12_mths_ex_med","acc_now_delinq","annual_inc")
dataset[replace_with_mean] <- lapply(dataset[replace_with_mean], function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  return(x)})
dataset[replace_with_median] <- lapply(dataset[replace_with_median], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)})
#Predictive impute for variable with high missingness
dataset2 <- dataset %>%
  select(tot_coll_amt)
dataset2$tot_coll_amt <- log1p(dataset2$tot_coll_amt)
dataset3 <- dataset %>%
  select(-tot_coll_amt)
datasetc <- cbind(dataset2,dataset3)
imputed_data <- mice(datasetc, method = "pmm", m = 5)
completed_data <- complete(imputed_data)
dataset_imp <- completed_data %>%
  select(tot_coll_amt,tot_cur_bal,total_rev_hi_lim)
dataset <- dataset %>%
  select(-tot_coll_amt,-tot_cur_bal,-total_rev_hi_lim)
dataset <- cbind(dataset,dataset_imp)
#Remove variables with high missingness
dataset <- dataset %>%
  select(-annual_inc_joint, -dti_joint, -open_acc_6m, -open_il_6m, -open_il_12m, -open_il_24m, -mths_since_rcnt_il, -total_bal_il, -il_util, -open_rv_12m, -open_rv_24m, -max_bal_bc, -all_util, -inq_fi, -total_cu_tl, -inq_last_12m)
#Introducing new missingness variables
dataset <- dataset %>%
  mutate(mths_since_last_delinq_missingness = ifelse(is.na(mths_since_last_delinq), 0, 1)) 
dataset <- dataset %>%
  mutate(mths_since_last_record_missingness = ifelse(is.na(mths_since_last_record), 0, 1))
dataset <- dataset %>%
  mutate(mths_since_last_major_derog_missingness = ifelse(is.na(mths_since_last_major_derog), 0, 1))
dataset <- dataset %>%
  select(-mths_since_last_delinq, -mths_since_last_record, -mths_since_last_major_derog)
#Recheck Missingness
colSums(is.na(dataset))


#Convert Date variables
#Selecting the relevant date variables
dataset2 <- dataset %>%
  select(issue_d,last_pymnt_d,next_pymnt_d,earliest_cr_line,last_credit_pull_d)
#splitting the date and month
dataset2$issue_d_m <- substr(dataset2$issue_d,1,3)
dataset2$issue_d_y <- substr(dataset2$issue_d,5,6)
dataset2$last_pymnt_d_m <- substr(dataset2$last_pymnt_d,1,3)
dataset2$last_pymnt_d_y <- substr(dataset2$last_pymnt_d,5,6)
dataset2$next_pymnt_d_m <- substr(dataset2$next_pymnt_d,1,3)
dataset2$next_pymnt_d_y <- substr(dataset2$next_pymnt_d,5,6)
dataset2$earliest_cr_line_m <- substr(dataset2$earliest_cr_line,1,3)
dataset2$earliest_cr_line_y <- substr(dataset2$earliest_cr_line,5,6)
dataset2$last_credit_pull_d_m <- substr(dataset2$next_pymnt_d,1,3)
dataset2$last_credit_pull_d_y <- substr(dataset2$next_pymnt_d,5,6)

#converting year in date to long form
dataset2$issue_d_y <- ifelse(as.numeric(dataset2$issue_d_y) < 26, 
                             paste0("20", dataset2$issue_d_y), 
                             paste0("19", dataset2$issue_d_y))
dataset2$last_pymnt_d_y <- ifelse(as.numeric(dataset2$last_pymnt_d_y) < 26, 
                                  paste0("20", dataset2$last_pymnt_d_y), 
                                  paste0("19", dataset2$last_pymnt_d_y))
dataset2$next_pymnt_d_y <- ifelse(as.numeric(dataset2$next_pymnt_d_y) < 26, 
                                  paste0("20", dataset2$next_pymnt_d_y), 
                                  paste0("19", dataset2$next_pymnt_d_y))
dataset2$earliest_cr_line_y <- ifelse(as.numeric(dataset2$earliest_cr_line_y) < 26, 
                                      paste0("20", dataset2$earliest_cr_line_y), 
                                      paste0("19", dataset2$earliest_cr_line_y))
dataset2$last_credit_pull_d_y <- ifelse(as.numeric(dataset2$last_credit_pull_d_y) < 26, 
                                      paste0("20", dataset2$last_credit_pull_d_y), 
                                      paste0("19", dataset2$last_credit_pull_d_y))
months_vector <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
dataset2$issue_d_m1 <- sprintf("%02d", match(dataset2$issue_d_m, months_vector))
dataset2$last_pymnt_d_m1 <- sprintf("%02d", match(dataset2$last_pymnt_d_m, months_vector))
dataset2$next_pymnt_d_m1 <- sprintf("%02d", match(dataset2$next_pymnt_d_m, months_vector))
dataset2$earliest_cr_line_m1 <- sprintf("%02d", match(dataset2$earliest_cr_line_m, months_vector))
dataset2$last_credit_pull_d_m1 <- sprintf("%02d", match(dataset2$last_credit_pull_d_m, months_vector))
#Concat final date string
dataset2$issue_d_f <- paste(dataset2$issue_d_y,dataset2$issue_d_m1, sep = "-", add = "01")
dataset2$last_pymnt_d_f <- paste(dataset2$last_pymnt_d_y,dataset2$last_pymnt_d_m1, sep = "-", add = "01")
dataset2$next_pymnt_d_f <- paste(dataset2$next_pymnt_d_y,dataset2$next_pymnt_d_m1, sep = "-", add = "01")
dataset2$earliest_cr_line_f <- paste(dataset2$earliest_cr_line_y,dataset2$earliest_cr_line_m1, sep = "-", add = "01")
dataset2$last_credit_pull_d_f <- paste(dataset2$last_credit_pull_d_y,dataset2$last_credit_pull_d_m1, sep = "-", add = "01")

#Selecting just the final date variables
dataset_d <- dataset2 %>%
  select(issue_d_f,last_pymnt_d_f,next_pymnt_d_f,earliest_cr_line_f,last_credit_pull_d_f)
#Replacing NA date cells with NA only
cleaned_data <- dataset_d %>%
  mutate(across(everything(), ~ ifelse(grepl("NA", ., ignore.case = TRUE), NA, .)))
# Find the most recent date across all date columns
max_date <- max(c(
  max(cleaned_data$issue_d_f, na.rm = TRUE),
  max(cleaned_data$last_pymnt_d_f, na.rm = TRUE),
  max(cleaned_data$next_pymnt_d_f, na.rm = TRUE),
  max(cleaned_data$earliest_cr_line_f, na.rm = TRUE),
  max(cleaned_data$last_credit_pull_d_f, na.rm = TRUE)
), na.rm = TRUE)

# Print the most recent date
print(max_date)


# Convert to Date format
cleaned_data <- cleaned_data %>%
  mutate(across(everything(), as.Date))

# Set base date
base_date <- as.Date("2016-03-01")

# Calculate duration in months
cleaned_data <- cleaned_data %>%
  mutate(across(everything(), ~ time_length(interval(base_date, .), "months")))
#Change duration to absolute values
cleaned_data <- cleaned_data %>% mutate(across(everything(), abs))
#Check for NAs
colSums(is.na(cleaned_data))
na_percentage <- colSums(is.na(cleaned_data)) / nrow(cleaned_data) * 100
na_percentage_2dp <- round(na_percentage,2)
print(na_percentage_2dp)
#Check Skewness for impute
skewness_values <- apply(cleaned_data, 2, function(x) skewness(as.numeric(x), na.rm = TRUE))
print(skewness_values)
#Impute median for columns with low missingness
replace_with_median2 <- c("last_pymnt_d_f","earliest_cr_line_f")
cleaned_data[replace_with_median2] <- lapply(cleaned_data[replace_with_median2], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)})
#Discard variables with high missingness
cleaned_data <- cleaned_data %>%
  select(-next_pymnt_d_f,-last_credit_pull_d_f)
#Combining date variables with main dataset + removing previous date variables
dataset <- dataset %>%
  select(-issue_d,-last_pymnt_d,-next_pymnt_d,-earliest_cr_line,-last_credit_pull_d)
df_combined <- cbind(dataset, cleaned_data)

#Check that categorical variables have the right class
unique(df_combined$application_type)

#Univariate analysis for continuous variables
numeric_data <- df_combined %>%
  select(-term,-grade,-sub_grade,-emp_length,-home_ownership,-verification_status,-loan_status,-pymnt_plan,-purpose,-addr_state,-initial_list_status,-application_type,-loan_status_binary,-mths_since_last_delinq_missingness,-mths_since_last_record_missingness,-mths_since_last_major_derog_missingness)
categorical_data <- df_combined %>%
  select(term,grade,sub_grade,emp_length,home_ownership,verification_status,loan_status,pymnt_plan,purpose,addr_state,initial_list_status,application_type,loan_status_binary,mths_since_last_delinq_missingness,mths_since_last_record_missingness,mths_since_last_major_derog_missingness)

#Looking at plots to identify skewed variables
hist(numeric_data$loan_amnt)
boxplot(numeric_data$loan_amnt)
#Compute skewness for variables
skewness_values <- apply(numeric_data, 2, function(x) skewness(as.numeric(x), na.rm = TRUE))
print(skewness_values)


#Looking through numeric columns for outliers
detect_outliers <- function(column) {
  Q1 <- quantile(column, 0.25, na.rm = TRUE)
  Q3 <- quantile(column, 0.75, na.rm = TRUE)
  IQR_value <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  sum(column < lower_bound | column > upper_bound, na.rm = TRUE)  # Count outliers
}

# Apply function to all numeric columns
outlier_counts <- sapply(numeric_data, detect_outliers)

# Print number of outliers per column
print(outlier_counts)
outlier_percentage <- outlier_counts/277140*100
print(outlier_percentage)
#Skew/Outlier Table
skew_outlier_df <- data.frame(
  Skew = skewness_values,
  Outlier_Percentage = outlier_percentage
)
rownames(skew_outlier_df) <- names(skewness_values)
#Deciding whether to winsorize and/or transform
skew_outlier_df <- skew_outlier_df %>%
  mutate(
    Winsorize = ifelse(outlier_percentage > 5, "Yes", "No"),
    Transform = ifelse(abs(Skew) > 1, "Yes", "No")
  )
#Winsorizing variables with more than 5% of outliers
cols_to_winsorize <- c("delinq_2yrs", "inq_last_6mths", "pub_rec","revol_bal","out_prncp","out_prncp_inv","total_rec_int","tot_coll_amt","total_rev_hi_lim","last_pymnt_d_f")
# Create a new dataframe to store the Winsorized data
winsorized_data <- numeric_data  # Copy original dataset
# Apply Winsorization only to selected columns
winsorized_data[cols_to_winsorize] <- lapply(winsorized_data[cols_to_winsorize], function(x) {
  lower <- quantile(x, 0.01, na.rm = TRUE)  # Compute 1st percentile
  upper <- quantile(x, 0.99, na.rm = TRUE)  # Compute 99th percentile
  pmax(pmin(x, upper), lower)  # Clamp values within range
})
#Calculate skewness after winsorizing
skew_after_winsor <- apply(winsorized_data, 2, function(x) skewness(as.numeric(x), na.rm = TRUE))
print(skew_after_winsor)
skew_outlier_df <- skew_outlier_df %>%
  mutate(skew_after_winsor = skew_after_winsor[rownames(skew_outlier_df)])
#Applying transformation to variables based on skew values
# Define the columns that still have skewness > 1 after Winsorization
cols_to_transform <- c("installment", "annual_inc","delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_bal","out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv","total_rec_int","total_rec_late_fee","last_pymnt_amnt","collections_12_mths_ex_med","acc_now_delinq","tot_coll_amt","tot_cur_bal","total_rev_hi_lim","last_pymnt_d_f","earliest_cr_line_f")  

# Apply transformations based on skewness
transformed_data <- winsorized_data  # Copy the dataset

for (col in colnames(transformed_data)) {
  
  # Calculate the skewness of the current column
  skew_val <- skewness(transformed_data[[col]], na.rm = TRUE)
  
 
    # If skewness is moderately high, apply log transformation
if (skew_val > 10) {  
    if (all(transformed_data[[col]] > 0, na.rm = TRUE)) {
      transformed_data[[col]] <- log(transformed_data[[col]] + 1)
    }
    
    # If skewness is mild, apply square root transformation
  } else if (skew_val > 1) {  
    transformed_data[[col]] <- sqrt(transformed_data[[col]])
  }
}
#calculate skewness after transforming 
skew_after_transform <- apply(transformed_data, 2, function(x) skewness(as.numeric(x), na.rm = TRUE))
print(skew_after_transform)
skew_outlier_df <- skew_outlier_df %>%
  mutate(skew_after_transform = skew_after_transform[rownames(skew_outlier_df)])

#Export skew table
write.csv(skew_outlier_df, "Skewness Table.csv", row.names = FALSE)
#Understanding variables that are still skewed
zeros <-(sum(winsorized_data$collections_12_mths_ex_med ==0))/277140*100
print(zeros)
#making highly skewed variables binary and combining them with categorical variables 
transformed_data <- transformed_data %>%
  select(-delinq_2yrs, -pub_rec, -out_prncp, -out_prncp_inv, -total_rec_late_fee, -tot_coll_amt, -collections_12_mths_ex_med, -acc_now_delinq)
skewed_data <- numeric_data %>%
  select( delinq_2yrs, pub_rec, out_prncp, out_prncp_inv, total_rec_late_fee, tot_coll_amt, collections_12_mths_ex_med, acc_now_delinq)
skewed_data <- skewed_data %>%
  mutate(
    delinq_2yrs_missingness = ifelse(delinq_2yrs > 0, 1, 0),
    pub_rec_missingness = ifelse(pub_rec > 0, 1, 0),
    out_prncp_missingness = ifelse(out_prncp > 0, 1, 0),
    out_prncp_inv_missingness = ifelse(out_prncp_inv > 0, 1, 0),
    total_rec_late_fee_missingness = ifelse(total_rec_late_fee > 0, 1, 0),
    tot_coll_amt_missingness = ifelse(tot_coll_amt > 0, 1, 0),
    collections_12_mths_ex_med_missingness = ifelse(collections_12_mths_ex_med > 0, 1, 0),
    acc_now_delinq_missingness = ifelse(acc_now_delinq > 0, 1, 0)
  )
missingness_columns <- skewed_data %>%
  select(
    delinq_2yrs_missingness,
    pub_rec_missingness,
    out_prncp_missingness,
    out_prncp_inv_missingness,
    total_rec_late_fee_missingness,
    tot_coll_amt_missingness,
    collections_12_mths_ex_med_missingness,
    acc_now_delinq_missingness
  )
# Add the missingness columns to categorical_data
categorical_data <- bind_cols(categorical_data, missingness_columns)

#Check for missingness in categorical data
missing_values <- sapply(categorical_data, function(x) sum(is.na(x)))
print(missing_values)
#
for (col in colnames(categorical_data)) {
  # Check if the column is categorical (either factor or character type)
  if (is.factor(categorical_data[[col]]) || is.character(categorical_data[[col]])) {
    
    # Create the plot
    p <- ggplot(categorical_data, aes(x = as.factor(categorical_data[[col]]))) + 
      geom_bar() +
      ggtitle(paste("Bar plot for", col)) + 
      xlab(col) + 
      ylab("Count") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    # Print the plot
    print(p)
  }
}

#Bivariate analysis
final_data <- cbind(transformed_data,categorical_data)
#Logistic regression with financial variables
summary(glm(loan_status_binary ~ loan_amnt, data = final_data, family = binomial))
summary(glm(loan_status_binary ~ funded_amnt_inv, data = final_data, family = binomial))
summary(glm(loan_status_binary ~ int_rate, data = final_data, family = binomial))
summary(glm(loan_status_binary ~ installment, data = final_data, family = binomial))
summary(glm(loan_status_binary ~ dti, data = final_data, family = binomial))
#Logistic regression with credit history variables 
summary(glm(loan_status_binary ~ open_acc, data = final_data, family = binomial))
summary(glm(loan_status_binary ~ revol_util, data = final_data, family = binomial))

#SMOTE required 

#Encoding categorical variables
unique(categorical_data$application_type)
#One-Hot Encoded variables
# Define the one-hot encoder for specific columns
encoder <- dummyVars(~ term + home_ownership + verification_status + pymnt_plan + initial_list_status + application_type, data = categorical_data)
# Apply encoding to selected columns
encoded_data <- predict(encoder, categorical_data)
# Convert to a data frame
encoded_data <- as.data.frame(encoded_data)
# Combine with the full dataset
cat1 <- cbind(encoded_data,categorical_data)
#Remove original columns of the encoded columns 
cat1 <- cat1 %>% 
  select(-term,-home_ownership,-verification_status,-pymnt_plan,-initial_list_status,-application_type)
#Ordinal Encoded variables
cat1$grade <- as.numeric(factor(cat1$grade, levels = c("A", "B", "C", "D", "E", "F","G")))
#Convert employment length to numeric
sum(cat1$emp_length == "n/a")
unique(cat1$emp_length)
cat1 <- cat1 %>%
  mutate(emp_length_num = case_when(
    emp_length == "less than 1 year" ~ 0,   # Manually set "less than 1 year" to 0
    emp_length == "1 year" ~ 1, # Manually set "1 year" to 1
    emp_length == "2 years" ~ 2,
    emp_length == "3 years" ~ 3,
    emp_length == "4 years" ~ 4,
    emp_length == "5 years" ~ 5,
    emp_length == "6 years" ~ 6,
    emp_length == "7 years" ~ 7,
    emp_length == "8 years" ~ 8,
    emp_length == "9 years" ~ 9,
    emp_length == "10 years" ~ 10,
    emp_length == "n/a" ~ NA_real_,  # Convert "n/a" to NA
    TRUE ~ as.numeric(emp_length)))  # Others remain as emp_length
median_value <- median(cat1$emp_length_num, na.rm = TRUE) #getting median value for imputation
print(median_value)
cat1 <- cat1 %>%
  mutate(emp_length_num = ifelse(is.na(emp_length_num), 4, emp_length_num))
cat1 <- cat1 %>% select(-emp_length)
#Drop first encoded variables
cat1 <- dummy_cols(cat1, select_columns = "purpose", remove_first_dummy = TRUE)
cat1 <- dummy_cols(cat1, select_columns = "addr_state", remove_first_dummy = TRUE)
#Remove purpose and addr_state
cat1 <- cat1 %>% select(-purpose,-addr_state) 
#sub-grade removed due to collinearity
cat1 <- cat1 %>% select(-sub_grade)

#applying feature scaling on transformed data
colMeans(transformed_data)
scaled_data <- scale(transformed_data)
colMeans(scaled_data)
numeric_df <- as.data.frame(scaled_data)

#Checking for collinearity
# Create a correlation matrix
correlation_matrix <- cor(numeric_df)
print(correlation_matrix)
# Visualize it using a heatmap
library(ggplot2)
library(reshape2)
melted_corr <- melt(correlation_matrix)
ggplot(melted_corr, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Identify pairs of variables with high correlation (>0.8 or <-0.8)
high_cor <- findCorrelation(correlation_matrix, cutoff = 0.8)
print(names(numeric_df)[high_cor])
correlated_pairs <- list()

# Loop through the correlation matrix to extract pairs of correlated variables
for (i in 1:(ncol(correlation_matrix) - 1)) {
  for (j in (i + 1):ncol(correlation_matrix)) {
    if (abs(correlation_matrix[i, j]) > 0.8) {
      # Add the pair of variables to the list
      correlated_pairs[[length(correlated_pairs) + 1]] <- c(colnames(correlation_matrix)[i], colnames(correlation_matrix)[j])
    }
  }
}

# Print the correlated pairs
print(correlated_pairs)
#Variables to remove
numeric_df_reduced <- numeric_df %>%
  select(-funded_amnt_inv,-total_pymnt_inv,-total_rec_prncp,-funded_amnt,-installment)

#Recheck correlation
correlation_matrix <- cor(numeric_df_reduced)
high_cor <- findCorrelation(correlation_matrix, cutoff = 0.8)
print(names(numeric_df_reduced)[high_cor])
correlated_pairs <- list()

# Loop through the correlation matrix to extract pairs of correlated variables
for (i in 1:(ncol(correlation_matrix) - 1)) {
  for (j in (i + 1):ncol(correlation_matrix)) {
    if (abs(correlation_matrix[i, j]) > 0.8) {
      # Add the pair of variables to the list
      correlated_pairs[[length(correlated_pairs) + 1]] <- c(colnames(correlation_matrix)[i], colnames(correlation_matrix)[j])
    }
  }
}

# Print the correlated pairs
print(correlated_pairs)

#Looking into collinearity in categorical variables
cat_col <- cat1 %>%
  select(-loan_status,-loan_status_binary)
cor_matrix <- cor(cat_col, use = "pairwise.complete.obs")
print(cor_matrix)
cor_df <- as.data.frame(as.table(cor_matrix))
high_cor_vars <- subset(cor_df, abs(Freq) > 0.8 & Var1 != Var2)
print(high_cor_vars)
#Remove collinear variables
cat_col <- cat_col %>%
  select(-'term 36 months',-home_ownershipRENT, -pymnt_plann, -initial_list_statusf, -application_typeJOINT,-mths_since_last_record_missingness,-out_prncp_inv_missingness)
#Recheck COllinearity
cor_matrix <- cor(cat_col, use = "pairwise.complete.obs")
print(cor_matrix)
cor_df <- as.data.frame(as.table(cor_matrix))
high_cor_vars <- subset(cor_df, abs(Freq) > 0.8 & Var1 != Var2)
print(high_cor_vars)

#Combining the dataset for modeling
cat_col <- cbind(cat_col, cat1 %>% select(loan_status, loan_status_binary))
df_final <- cbind(numeric_df_reduced,cat_col)
#Data for logistic regression 
df_lr <- df_final %>%
  select(-loan_status)
#Logistic Regression Model
set.seed(123) #fixing a code for randomness so the same code can be used to replicate
split = sample.split(df_lr$loan_status_binary,SplitRatio=0.8)

training_set = subset(df_lr,split == TRUE)
test_set = subset(df_lr,split == FALSE)
#Applying SMOT
# Convert loan_status_binary to a factor
training_set$loan_status_binary <- as.factor(training_set$loan_status_binary)

# Apply SMOTE
set.seed(123)  # For reproducibility

smote_result <- SMOTE(training_set[, !names(training_set) %in% "loan_status_binary"], 
                      training_set$loan_status_binary, 
                      K = 3, dup_size = 1)

# Combine the SMOTE result into a new data frame
training_set_smote <- smote_result$data
colnames(training_set_smote)[ncol(training_set_smote)] <- "loan_status_binary"

# Convert loan_status_binary back to a factor
training_set_smote$loan_status_binary <- as.factor(training_set_smote$loan_status_binary)

# Check the distribution of the target variable after SMOTE
summary(training_set_smote$loan_status_binary)

# Check the class distribution before and after SMOTE
table(training_set$loan_status_binary)  # Before SMOTE
table(training_set_smote$loan_status_binary)  # After SMOTE

#TESTTEST LR
# Fitting the Logistic Regression Model to training_set
classifier = glm(formula = loan_status_binary~.,
                 family = binomial,
                 data = training_set_smote)

#troubleshooting
summary(training_set_smote)
training_set_smote %>%
  group_by(loan_status_binary) %>%
  summarise_all(~ length(unique(.)))
# Identify variables with only one unique value for either class
problematic_vars <- training_set_smote %>%
  group_by(loan_status_binary) %>%
  summarise_all(~ length(unique(.))) %>%
  pivot_longer(-loan_status_binary, names_to = "variable", values_to = "unique_values") %>%
  filter(unique_values == 1) %>%
  pull(variable)
# Print the problematic variables
print(problematic_vars)
#Remove problematic vars
training_set_smote <- training_set_smote %>%
  select(-home_ownershipANY,-addr_state_ME, -application_typeINDIVIDUAL,-out_prncp_missingness)

#Applying lasso regression
model_lasso <- cv.glmnet(as.matrix(training_set_smote[, -which(names(training_set_smote) == "loan_status_binary")]), 
                         training_set_smote$loan_status_binary, 
                         family = "binomial", alpha = 1)  # Lasso
# Convert test set to matrix (excluding target variable)
X_test <- as.matrix(test_set[, -which(names(test_set) == "loan_status_binary")])
# Predict probabilities
y_pred_prob <- predict(model_lasso, newx = X_test, s = "lambda.min", type = "response")
y_pred <- as.vector(predict(model_lasso, newx = X_test, s = "lambda.min", type = "response"))
# Convert to binary predictions (threshold = 0.5)
y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)
# Combine actual vs predicted values into a data frame
df <- data.frame("Actual" = test_set$loan_status_binary, "Predicted" = y_pred)
#Plot predicted vs. actual
ggplot(df, aes(x = Actual, y = Predicted, color = factor(Actual == round(Predicted)))) +
  geom_jitter(width = 0.2, height = 0.2, alpha = 0.5) +
  scale_color_manual(values = c("TRUE" = "blue", "FALSE" = "red")) + 
  theme_minimal() +
  labs(title = "Actual vs Predicted Loan Status (Logistic Regression)",
       x = "Actual Loan Status",
       y = "Predicted Loan Status",
       color = "Correct Prediction?")
#Making the Confusion Matrix 
conf_matrix <- table(Actual = df$Actual, Predicted = df$lambda.min)
print(conf_matrix)

#Recall for Logistic regression
recall <- conf_matrix[2,2] / sum(conf_matrix[2,])  # TP / (TP + FN)
print(recall)

#AUC-ROC for Logistic Regression
roc_curve <- roc(df$Actual, df$lambda.min)  # Use predicted probabilities
auc(roc_curve)

#Understanding the features 
coef(model_lasso)


#Applying Random Forest
colnames(training_set_smote)[which(colnames(training_set_smote) == "term 60 months")] <- "term_60_months"
colnames(training_set_smote)[which(colnames(training_set_smote) == "verification_statusNot Verified")] <- "verification_statusNot_Verified"
colnames(training_set_smote)[which(colnames(training_set_smote) == "verification_statusSource Verified")] <- "verification_statusSource_Verified"
model_rf_binary <- randomForest(loan_status_binary ~ ., data = training_set_smote, ntree = 100)
colnames(test_set)[which(colnames(test_set) == "term 60 months")] <- "term_60_months"
colnames(test_set)[which(colnames(test_set) == "verification_statusNot Verified")] <- "verification_statusNot_Verified"
colnames(test_set)[which(colnames(test_set) == "verification_statusSource Verified")] <- "verification_statusSource_Verified"
predictions <- predict(model_rf_binary, newdata = test_set)
test_set$loan_status_binary <- factor(test_set$loan_status_binary, levels = c("0", "1"))
#Recall (Specificity)
cm <- confusionMatrix(predictions, test_set$loan_status_binary)
recall <- cm$byClass["Sensitivity"]
print(recall)
#AUC-ROC
pred_prob <- predict(model_rf_binary, newdata = test_set, type = "prob")[,2] 
# Compute ROC Curve
roc_obj <- roc(test_set$loan_status_binary, pred_prob)
# Plot ROC Curve
plot(roc_obj, col = "blue", main = "ROC Curve - Random Forest")
# Compute AUC
auc_value <- auc(roc_obj)
print(paste("AUC:", auc_value))

#XGBoost Model 
# Convert categorical variables to factors
training_set_smote$loan_status_binary <- as.numeric(as.factor(training_set_smote$loan_status_binary)) - 1
test_set$loan_status_binary <- as.numeric(as.factor(test_set$loan_status_binary)) - 1

# Convert dataset to XGBoost matrix format
training_matrix <- as.matrix(training_set_smote[, -which(names(training_set_smote) == "loan_status_binary")])
test_matrix <- as.matrix(test_set[, -which(names(test_set) == "loan_status_binary")])

# Define target variable
train_label <- training_set_smote$loan_status_binary
test_label <- test_set$loan_status_binary
#Model
model_xgb <- xgboost(data = training_matrix, 
                     label = train_label, 
                     nrounds = 100,  # Number of boosting rounds
                     objective = "binary:logistic",  # Binary classification
                     eval_metric = "auc",  # Use AUC as evaluation metric
                     verbose = 1)
#Making predictions 
pred_prob_xgb <- predict(model_xgb, test_matrix)  # Probabilities
pred_xgb <- ifelse(pred_prob_xgb > 0.5, 1, 0)  # Convert to binary labels
#Confusion matrix
conf_matrix_xgb <- confusionMatrix(factor(pred_xgb), factor(test_label))
print(conf_matrix_xgb)
#AUC-ROC 
roc_obj_xgb <- roc(test_label, pred_prob_xgb)
auc_value_xgb <- auc(roc_obj_xgb)
print(paste("AUC:", auc_value_xgb))
#Recall 
recall_xgb <- conf_matrix_xgb$byClass["Sensitivity"]
print(paste("Recall:", recall_xgb))

#Feature Analysis
importance_matrix <- xgb.importance(model = model_xgb)
print(importance_matrix)