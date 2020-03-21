#### Step 1 : DXYArea_selected.rds ####
setwd("D:/zxy/Study/Project/Wuhan")
rm(list=ls())
# 1. Download data from https://github.com/BlankerL/DXY-COVID-19-Data/tree/master/csv
# 2. Load data from csv and clean
data = read.csv("data/DXYArea.csv", encoding="UTF-8", stringsAsFactors=FALSE)
# Reorder data by province _ city _ then update time
data = data[  order( data[,3],data[,6], data[,15] ),  ]
names(data)[15]
time = do.call(rbind, strsplit(data[,15], ' ')) # updateTime
# data[,15] = time[,1]
# data$date = time[,2]
data <- cbind(data, date=as.Date(time[,1], "%Y-%m-%d"))
data = data[order(data[,3],data[,6],data[,16], data[,15],decreasing = T),]
confirm_index = c(3,6,16)
names(data)[confirm_index]


data = removesamedatedata(data)
head(data)
#saveRDS(data,"data/DXYArea_selected.rds")

data=readRDS("data/DXYArea_selected.rds")
head(data)
# data[data$provinceEnglishName=='Beijing',]
cat("Time range of data")
print(range(data$date))
unique(data$cityName)[1:10]
sort( unique(data$cityName) )[1:10]
sort( unique(data$cityEnglishName) )[1:10]

# Compare city names
#
# head(gdp2018)
# library(Hmisc)
# capitalize(as.character(gdp2018$city[1]))
#
#
# newcity = c()
# for(i in 1:50){
#   newcity[i] = capitalize(as.character(gdp2018$city[i]))
# }
# gdp2018$city = newcity
#
#
# # Observe xian missing value
# which(gdp2018$city=="Xi'an")
# data[ data$date=="2020-03-02" & data$cityEnglishName == "Xi'an",]
# data$cityEnglishName
#
#
#
# data[ data$provinceEnglishName=="yunnan",]
# data2[,c(2,5,11:14)]
# output数据sample，方便第一步选择城市
# data3 = cbind(data2$X.U.FEFF.provinceName,data2$cityName, data2$city_confirmedCount,
# data2$city_curedCount, data2$city_deadCount)
# colnames(data3) <- c("Province", "City", "#Confirm","#Cured","#Dead")
# data[1,]
#
# # city with top confirm
# data3

#### Step 2: qianxi_ratio.csv ####
rm(list=ls())
library(xzhang97wuhan)
qianxi <- read.csv("data/qianxi.csv", header = T)[,-1]

cities = names(qianxi)[-c(1:2)]
qianxi_ratio = dataCleanQianxiRatio()
#names(qianxi_ratio)
#write.csv(qianxi_ratio, "data/qianxi_ratio.csv")
cbind((qianxi[24:26,1:5]), (qianxi[100:102,2:5]))
head(qianxi_ratio[,1:6],3)
plotQianxiRatioAll()

#### Step 3: ratio.txt correlation between action and qianxi_ratio ####
action = read.csv("data/action.csv", skip=4)[,-1]
qianxi_ratio = read.csv("data/qianxi_ratio.csv", header = T)[,-1]
ratio = generateCorrelationBetweenQianxiratioAndAction()
write.table(ratio, "data/ratio.txt")


#### Step 4: action and qianxi plot ####
rm(list=ls())
setwd("D:/zxy/Study/Project/Wuhan")
library(xzhang97wuhan)
action  = read.csv("data/action.csv", skip=4)
city="Wuhan"
city_action = action[,which(names(action)=="Wuhan")]
qianxi <- read.csv("data/qianxi.csv", header = T)[,-1]
names(qianxi)
length(qianxi$wuhanshi)
plotWuhanQianxi()
# plot qianxi.pdf
plotQQianxiStrengthPdf()
# generate future qianxi
plotWuhanQianxiPlosAction()
# plot qianxi and action
action  = read.csv("data/action.csv", skip=4)
qianxi <- read.csv("data/qianxi.csv", header = T)[-1,-1]
plotWuhanQianxi("wuhan")
plotQianxiwithAction(1)
r = read.table("data/ratio.txt")
plotQQianxiWithActionPdf()


#### Step 5 Combine data together ####
rm(list=ls())
library(xzhang97wuhan)
data = readRDS("data/DXYArea_selected.rds");head(data)
gdp2018 = read.csv("data/2018gdp.csv", header=T)[,-c(1,4)];head(gdp2018)
action  = read.csv("data/action.csv", skip=4)[,-1];head(action)
qianxi = read.csv("data/qianxi.csv", header = T, skip=1)[,-1];head(qianxi)
qianxi_ratio = read.csv("data/qianxi_ratio.csv", header = T, skip=1)[,-1];head(qianxi_ratio)
city_names_infile = read.csv("data/city_name_DXY.csv", header=T)[1:50,-1];head(city_names_infile)
#state = readRDS("data/covid.rds")
tl_index = c(1,1,1,1,2,2,2,2,2)
#generateStateOfACity(42,"Wuhan")
state=generateStateOfAllCities(42)
tianjin = state[['Tianjin']]
saveRDS(state, file = "data/covid.rds")

#### Step 6: final step: data overview ####
rm(list=ls())
library(xzhang97wuhan)
state = readRDS("data/covid.rds")
summary(state)
city = state[['Tianjin']]
str(city)











