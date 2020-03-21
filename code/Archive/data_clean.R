# You can learn more about package authoring with RStudio at:
#
#   http://r-pkgs.had.co.nz/
#
# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'
generateStateOfACity <- function(T=42, c="Tianjin"){
  state <- list()
  # load data from DXY by city and province zip code
  dates = sort(unique(data$date), increasing=T)[1:T]
  zip = city_names_infile$city_zipCode[which( city_names_infile$city == c )]
  province=(is.na(zip))
  if (!province){
    data_city = data[which(data$city_zipCode == zip), ] #gdp2018$city[c]
    head(data_city)
  } else {
    zip = city_names_infile$province_zipCode[which( city_names_infile$city == c )]
    data_city = data[ data$province_zipCode == zip, ]
    data_city = data_city[order(data_city$date),]
    data_city = removesamedatedata(data_city, index=c(3,16))
  }

  data_city = data_city[order(data_city$date),]
  city_data <- data_city[c(1:T),]
  correct_index = which( dates %in% data_city$date )
  city_data[correct_index,] = data_city[which( data_city$date %in% dates),]

  wrong_index = which( c(1:T) %in% correct_index == FALSE )
  print(c)
  print(wrong_index)

  # Initialization for the beginning missing data
  i=1
  while(i %in% wrong_index){
    city_data[i,] = city_data[correct_index[1],]
    city_data[i,'date'] = dates[i]
    city_data[i,11:14]=0
    i=i+1
  }
  t_prime_l = rep(i, T)
  # Backward imputation for other than the first date
  for(j in i:T){
    if(j %in% wrong_index){
      city_data[j,] = city_data[j-1,]
      city_data[j,'date'] = dates[j]
      city_data[j,11:14] = floor((city_data[j-1,11:14] + city_data[j+1,11:14])/2)
    }
  }

  # load population and gdp from gdp2018
  index_in_gdp=which(gdp2018$city == c)
  gdp = rep(gdp2018$GDP[index_in_gdp], T)
  population = gdp2018$Population.10.000.[index_in_gdp]*1e4

  # Calculate three state variables
  if(province){
    infected <- city_data$province_confirmedCount/(population)
    remove <- (city_data$province_curedCount+city_data$province_deadCount)/(population)
  } else {
    infected <- city_data$city_confirmedCount/(population)
    remove <- (city_data$city_curedCount+city_data$city_deadCount)/(population)
  }

  suspect <- 1-infected-remove
  city_action = action[,which(names(action) == c)]
  m_qianxi2020 <- qianxi[c(1:T)+99, which(names(qianxi) == c)]
  m_qianxi2019 <- qianxi[c(1:T)+99-76, which(names(qianxi) == c)]
  ratio19to20 <- qianxi_ratio[c(1:T), which(names(qianxi_ratio) == c)]
  popu <- rep(population,T)
  # include qianxi m and ratio r
  state <- data.frame(dates, suspect, infected, remove, action=city_action,
                      m19 =m_qianxi2019 , m20=m_qianxi2020, ratio=ratio19to20,
                      gdp,popu=popu, tl = t_prime_l )
  return(state)
}


generateStateOfAllCities <- function(T=42){
  dates = sort(unique(data$date))[1:T]
  cities = gdp2018$city
  state <- list()
  i=1
  city=( cities[1])
  for (city in cities){
    state[[i]] <- generateStateOfACity(T, as.character( city))
    i=i+1
  }
  names(state)=cities
  return(state)
}
#
# generateStateOfACityInProvince <- function(data=data, T=42, c="Chaoyang District"){
#   state <- list()
#   dates = sort(unique(data$date), decreasing = FALSE)[1:T]
#   data_city = data[ data$cityEnglishName == c, ] #gdp2018$city[c]
#   city_data <- data_city[1:T,]
#   i=1
#   j=1
#   city_data[i,] = data_city[data_city$date==dates[j],][1,]
#   while(is.na(data_city[data_city$date==dates[j],][1,1])){
#     city_data[i,] = data_city[data_city$date==dates[j+1],][1,]
#     j=j+1
#   }
#
#   city_data[i,]
#   for(i in 2:T){
#     if(is.na(data_city[data_city$date==dates[i],][1,1])){
#       city_data[i,] = city_data[i-1,]
#     } else{
#       city_data[i,] = data_city[data_city$date==dates[i],][1,]
#     }
#   }
#   c_index = which(gdp2018$city == "Beijing")
#   population <- gdp2018$Population[c_index]
#   infected <- city_data$city_confirmedCount/(population*10000)
#   remove <- (city_data$city_curedCount+city_data$city_deadCount)/(population*10000)
#   #suspect <- 1-infected-remove
#   #gdp = rep(gdp2018$GDP[c_index], T)
#   state <- data.frame(dates, infected, remove)
#   return(state)
# }
#
# sumprov <- function(time, column=2){
#   temp=0
#   for (i in 1:l){
#     temp = temp +  province[[i]][time,column]
#   }
#   return(temp)
# }
#
# provinceGenerateState <- function(c=16, T=42){
#   # beijing c=16, c_index=c(1:6,9:18)
#   dates = sort(unique(data$date))[1:T]
#   data_province = data[ data$provinceEnglishName == gdp2018$city[c], ]
#   #city_data <- data_province[1:T,]
#   cities = unique(data_province$cityEnglishName)
#   province <- list()
#   l = length(cities)
#   for (i in 1:l){
#     province[[i]] <- generateStateOfACityInProvince(data=data, T=42, c=cities[i])
#   }
#   t=length(dates)
#   province_infect <- c()
#   for(time in 1:t){
#     province_infect[time] <- sumprov(time, 2)
#   }
#   province_remove <- c()
#   for(time in 1:t){
#     province_remove[time] <- sumprov(time, 3)
#   }
#   province_all  <- province[[1]][1:T,1:2]
#   province_all$infected = province_infect[1:T]
#   province_all$remove = province_remove[1:T]
#   return(province_all)
# }
#
#
# generateStateOfAll50Cities <- function(T=42){
#   state <- list()
#   dates = sort(unique(data$date))[1:T]
#   for (c in c(1:15)){
#     state[[c]] <- generateStateOfACity(data, T=42, c=gdp2018$city[c])
#   }
#   for (c in c(16:19)){
#     state[[c]] <- provinceGenerateState(c)
#   }
#   for (c in c(20:50)){
#     state[[c]] <- generateStateOfACity(data, T=42, c=gdp2018$city[c])
#   }
#   names(state)=gdp2018$city
#   return(state)
# }
#
# # 确认省是对的
# generateCorrelationBetweenQianxiratioAndAction <- function(){
#   action = read.csv("data/action.csv", skip=4)[,-1]
#   qianxi_ratio = read.csv("data/qianxi_ratio.csv", header = T)[,-1]
#   head(qianxi_ratio[,4:5])
#   cities = names(qianxi_ratio)[4:53]
#   n = dim(action)[1]; n
#   ratio <- c()
#   for(i in c(1:50)){
#     ratio[i]= cor(qianxi_ratio[1:n,i+3], action[, i+1])
#   }
#   ratio = cbind(cities, ratio = ratio)
#   return(ratio)
# }
#



removesamedatedata <- function(data, index=c(3,6,16)){
  repetative_index = c()
  for(i in 2:dim(data)[1]){
    if(sum(data[i,index] == data[i-1,index] , na.rm = T)==length(index)){
      repetative_index = rbind(repetative_index,i)
    }
  }
  data = data[-repetative_index,]
  return(data)
}






