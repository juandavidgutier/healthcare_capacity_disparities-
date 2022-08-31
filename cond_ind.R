#script to reproduce the results of the conditional independence of the paper:
#"Impact of healthcare capacity disparities in the COVID-19 vaccination 
#coverage in the United States: A cross-sectional study"




library(tidyverse)
library(broom)
library(dagitty)
library(lavaan)
library(ggdag)
library(ggtext)   
library(CondIndTests)
library(dplyr)
library(GGally)


data <- read.csv("D:/clases/UDES/articulo CI Diego/Datasets/dataset_covid.csv")
data <- na.omit(data)
str(data)

#ranks
quantile(data$SVI, na.rm=T, c(0:3/3))
data <- data %>% mutate(SVI = case_when(SVI < 0.31 ~ 1, 
                                        SVI >= 0.31 & SVI < 0.63  ~ 2, 
                                        SVI >= 0.63  ~ 3))


quantile(data$Vaccine_hesitancy, na.rm=T, c(0:3/3))
data <- data %>% mutate(Vaccine_hesitancy = case_when(Vaccine_hesitancy < 0.1696 ~ 1, 
                                                      Vaccine_hesitancy >= 0.1696 & Vaccine_hesitancy < 0.2136  ~ 2, 
                                                      Vaccine_hesitancy >= 0.2136  ~ 3))


quantile(data$RCHSI, na.rm=T, c(0:3/3))
data <- data %>% mutate(RCHSI = case_when(RCHSI < 0.2705083 ~ 1, 
                                          RCHSI >= 0.2705083 & RCHSI < 0.5560862 ~ 2, 
                                          RCHSI >= 0.5560862  ~ 3))


quantile(data$HACBI, na.rm=T, c(0:3/3))
data <- data %>% mutate(HACBI = case_when(HACBI < 0.3054229 ~ 1, 
                                          HACBI >= 0.3054229 & HACBI < 0.6055396 ~ 2, 
                                          HACBI >= 0.6055396  ~ 3))



data_dag <- dplyr::select(data, low_vaccination_rate, SVI , Vaccine_hesitancy, RCHSI, HACBI)

data_dag <- as.data.frame(data_dag)
str(data_dag)




#DAG0_5 OJO ESTA ES UNA VARIACION DEL DAG0_5 PARA INCLUIR HESITANT
g <- dagitty('dag {

HACBI [pos="0,-1"]
SVI  [pos="1,-1"]
RCHSI [pos="1,0.5"]
Vaccination_rate_.50. [pos="-1,0.5"]
Vaccine_hesitancy [pos="-1,-0.7"]

SVI -> RCHSI
SVI -> Vaccination_rate_.50.
                    
HACBI -> RCHSI
HACBI -> Vaccination_rate_.50.
                    
SVI -> HACBI
                    
SVI -> Vaccine_hesitancy
HACBI -> Vaccine_hesitancy

RCHSI -> Vaccine_hesitancy
                                                   
RCHSI -> Vaccination_rate_.50.
                    
Vaccine_hesitancy -> Vaccination_rate_.50.


              }')  

plot(g)

## Independencias condicionales
impliedConditionalIndependencies(g) #no conditional independence identified



