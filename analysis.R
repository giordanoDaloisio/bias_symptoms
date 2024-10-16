library(tidyverse)
library(Hmisc)

data <- read_csv("replication_package/data/bias_symptoms.csv")
view(data)

data <- subset(data, select = -c(variable, data))
head(data)

v <- varclus(
  as.matrix(data, similarity = "spearman", type = "data.matrix")
)

plot(v)

corr <- cor(data, method = "spearman")
view(corr)

m <- redun(as.matrix(data, r2 = 0.8, nk = 0))
reducedData <- m$In
print(reducedData)

corr <- cor(reducedData, method = "spearman")
view(corr)
