setwd("~/Github/2020-Postech-Open-innovation-Bigdata-Challenge")

fac2chr <- function(data){
  i <- sapply(data, is.factor)
  data[i] <- lapply(data[i], as.character)
  return(data)
}

data_t_n<-fac2chr(read.csv(file="daily_temperatures_national.csv"))
data_t_p<-fac2chr(read.csv(file="daily_temperatures_pohang.csv"))
data_t_s<-fac2chr(read.csv(file="daily_temperatures_seoul.csv"))

data_p_n<-fac2chr(read.csv(file="daily_precipitation_national.csv"))
data_p_p<-fac2chr(read.csv(file="daily_precipitation_pohang.csv"))
data_p_s<-fac2chr(read.csv(file="daily_precipitation_seoul.csv"))

data_t<-rbind(data_t_n, data_t_p, data_t_s)
data_p<-rbind(data_p_n, data_p_p, data_p_s)
data_p[is.na(data_p)] <- 0

data=merge(data_t, data_p, by=c("날짜","지점"))

data["날짜"]<-sapply(data["날짜"], as.Date)
      