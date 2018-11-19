library(dplyr)
library(ggplot2)
library(Hmisc)
library(survival)
library(rms)
library(feather)

#Parse SUPPORT trial data from Vanderbilt Biostatistics web site
#If lab value missing, set to recommended value in: http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc?sortcol=1;table=1;up=0#sorted_table
#We do not include predictor variables if missing in >4000 patients

data_dir <- '/Users/michael/Documents/research/machine learning/prognosis/neural\ network\ survival/data/'
figure_dir <- '/Users/michael/Documents/research/machine learning/prognosis/neural\ network\ survival/figures/'

data <- read.csv(paste(data_dir,'support2.csv',sep=''),stringsAsFactors=FALSE)
data$edu[is.na(data$edu)] <- median(data$edu,na.rm=T)
data$scoma[is.na(data$scoma)] <- median(data$scoma,na.rm=T)
data$avtisst[is.na(data$avtisst)] <- median(data$avtisst,na.rm=T)
data$sps[is.na(data$sps)] <- median(data$sps,na.rm=T)
data$aps[is.na(data$aps)] <- median(data$aps,na.rm=T)
data$meanbp[is.na(data$meanbp)] <- median(data$meanbp,na.rm=T)
data$wblc[is.na(data$wblc)] <- 9
data$hrt[is.na(data$hrt)] <- median(data$hrt,na.rm=T)
data$resp[is.na(data$resp)] <- median(data$resp,na.rm=T)
data$temp[is.na(data$temp)] <- median(data$temp,na.rm=T)
data$pafi[is.na(data$pafi)] <- 333.3 
data$alb[is.na(data$alb)] <- 3.5
data$bili[is.na(data$bili)] <- 1.01
data$crea[is.na(data$crea)] <- 1.01
data$sod[is.na(data$sod)] <- median(data$sod,na.rm=T)
data$ph[is.na(data$ph)] <- median(data$ph,na.rm=T)
data$adls[is.na(data$adls)] <- median(data$adls,na.rm=T)
data$income[data$income==''] <- 'unknown'
data$race[data$race==''] <- 'unknown'

myFormula <- formula(~d.time+death+dzgroup+num.co+edu+income+scoma+avtisst+race+sps+aps+hday+diabetes+dementia+ca+meanbp+wblc+hrt+resp+
                       temp+pafi+alb+bili+crea+sod+ph+adls)
data2 <- model.matrix(myFormula,data=data)
data2 <- data2[,-1]
colnames(data2) <- c("time","dead","dzgroupCHF","dzgroupCirrhosis","dzgroupColonCancer","dzgroupComa","dzgroupCOPD",
                     "dzgroupLungCancer","dzgroupMOSFMalig","comorbs","edu","income11to25","income25to50",
                     "incomeunder11","incomeunknown","scoma","avtisst","raceblack",
                     "racehispanic","raceother","raceunknown","racewhite","sps","aps","hday","diabetes","dementia",
                     "cano","cayes","meanbp","wblc","hrt","resp","temp","pafi","alb","bili","crea","sod","ph","adls")
write.table(data2,file=paste(data_dir,'support_parsed.csv',sep=''),row.names=FALSE,sep=',')
