rm(list=ls())

#TP de Alejandro Nieto, Giuse Macri, Ezequiel Lerner

library(xgboost)
library(corpus)
library(NLP)
library(tm)
library(dplyr)
library(gsubfn)
library(ggplot2)
setwd("C:\\LocalStorage")

one_hot_sparse <- function(data_set) {

    # IMPORTANTE: si una de las variables es de fecha, la va a ignorar

    require(Matrix)
    created <- FALSE

    if (sum(sapply(data_set, is.numeric)) > 0) {  # Si hay, Pasamos los numéricos a una matriz esparsa (sería raro que no estuviese, porque "Price"  es numérica y tiene que estar sí o sí)
        out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.numeric)]), "dgCMatrix")
        created <- TRUE
    }

    if (sum(sapply(data_set, is.logical)) > 0) {  # Si hay, pasamos los lógicos a esparsa y lo unimos con la matriz anterior
        if (created) {
            out_put_data <- cbind2(out_put_data,
                                   as(as.matrix(data_set[,sapply(data_set, is.logical)]), "dgCMatrix"))
        } else {
            out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.logical)]), "dgCMatrix")
            created <- TRUE
        }
    }

    # Identificamos las columnas que son factor (OJO: el data.frame no debería tener character)
    fact_variables <- names(which(sapply(data_set, is.factor)))

    # Para cada columna factor hago one hot encoding
    i <- 0

    for (f_var in fact_variables) {

        f_col_names <- levels(data_set[[f_var]])
        f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
        j_values <- as.numeric(data_set[[f_var]])  # Se pone como valor de j, el valor del nivel del factor
        
        if (sum(is.na(j_values)) > 0) {  # En categóricas, trato a NA como una categoría más
            j_values[is.na(j_values)] <- length(f_col_names) + 1
            f_col_names <- c(f_col_names, paste(f_var, "NA", sep = "_"))
        }

        if (i == 0) {
            fact_data <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                      x = rep(1, nrow(data_set)),
                                      dims = c(nrow(data_set), length(f_col_names)))
            fact_data@Dimnames[[2]] <- f_col_names
        } else {
            fact_data_tmp <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                          x = rep(1, nrow(data_set)),
                                          dims = c(nrow(data_set), length(f_col_names)))
            fact_data_tmp@Dimnames[[2]] <- f_col_names
            fact_data <- cbind(fact_data, fact_data_tmp)
        }

        i <- i + 1
    }

    if (length(fact_variables) > 0) {
        if (created) {
            out_put_data <- cbind(out_put_data, fact_data)
        } else {
            out_put_data <- fact_data
            created <- TRUE
        }
    }
    return(out_put_data)
}

## Cargo los datos

data_set <- readRDS("training_set.RDS")


## Hago bow model

corpus <- VCorpus(VectorSource(data_set$description))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, content_transformer(removePunctuation))
dt.mat <- DocumentTermMatrix(corpus,
                             control=list(stopwords=TRUE,
                                          wordLengths=c(0, 18),
                                          bounds=list(global=c(70,Inf))))
X_bow <- Matrix::sparseMatrix(i=dt.mat$i, 
                              j=dt.mat$j, 
                              x=dt.mat$v, 
                              dims=c(dt.mat$nrow, dt.mat$ncol),
                              dimnames = dt.mat$dimnames)

#Histograma del precio

ggplot(data_set, aes(x = price, fill = ..count..)) +
    geom_histogram(binwidth = 5000) +
    ggtitle("Figure 1 Histogram of price") +
    ylab("Count of houses") +
    xlab("Housing Price") + 
    theme(plot.title = element_text(hjust = 0.5)) + xlim(0,1000000)


## Genero nuevas variables, algunas no las ocupo y las dejo como comentario.

data_set$price <- log(data_set$price + 1)  # Voy a predecir el logaritmo de los precios
#data_set$pozo<- as.numeric(grepl("pozo",data_set$description, ignore.case = TRUE))
#data_set$estrenar<- as.numeric(grepl("estrenar",data_set$description, ignore.case = TRUE))
data_set$monoambiente<- as.numeric(grepl("monoambiente",data_set$description, ignore.case = TRUE))
#data_set$balcon<- as.numeric(grepl("balcon",data_set$description, ignore.case = TRUE))
#data_set$duplex<- as.numeric(grepl("duplex",data_set$description, ignore.case = TRUE))
data_set$obra<- as.numeric(grepl("construcción|obra",data_set$description, ignore.case = TRUE))
data_set$pileta <- as.numeric(grepl("pileta|piscina|alberga",data_set$description, ignore.case = TRUE))
#data_set$galpon<- as.numeric(grepl("galpon",data_set$description, ignore.case = TRUE))
#data_set$country<- as.numeric(grepl("country",data_set$description, ignore.case = TRUE))
#data_set$quinta<- as.numeric(grepl("quinta",data_set$description, ignore.case = TRUE))
data_set$barrioprivado<- as.numeric(grepl("barrio cerrado|barrio privado",data_set$description, ignore.case = TRUE))
#data_set$esquina<- as.numeric(grepl("esquina",data_set$description, ignore.case = TRUE))
data_set$remodelar<- as.numeric(grepl("refaccionar",data_set$description, ignore.case = TRUE))
data_set$singarantia <- as.numeric(grepl("sin garantia",data_set$description, ignore.case = TRUE))
data_set$industrial<- as.numeric(grepl("parque industrial",data_set$description, ignore.case = TRUE))
data_set$end_date <- ifelse(data_set$end_date < 600000, data_set$end_date, NA)
data_set$tiempo <- as.numeric(data_set$start_date - data_set$end_date)
data_set$latlon2 <- data_set$lat + data_set$lon
data_set$latlon3 <- data_set$lat - data_set$lon
data_set$latlon4 <- data_set$lon/data_set$lat
data_set$latlon5 <- data_set$lat+data_set$lat + data_set$lon
data_set$latlon6 <- data_set$lat*2 + data_set$lon*data_set$lon
data_set$latlon7 <- data_set$lat*data_set$lat + data_set$lon*data_set$lon
data_set$latlon9 <- data_set$lat*data_set$lat*-60 + data_set$lon*data_set$lon
data_set$lat2 <- data_set$lat * data_set$lat
data_set$lon2 <- data_set$lon * data_set$lon

#Hago los gráficos

ggplot(data_set, aes(surface_covered, price)) + geom_point(color = 'blue') + theme_bw() + xlim(0,10000)

ggplot(data_set, aes(x = price, fill = ..count..)) +
    geom_histogram(binwidth = 0.5) +
    ggtitle("Figure 1 Histogram of price") +
    ylab("Count of houses") +
    xlab("Housing Price") + 
    theme(plot.title = element_text(hjust = 0.5))


ggplot(data_set, aes(price)) +
    geom_histogram(aes(fill = property_type), position = position_stack(reverse = TRUE), binwidth = 0.5) +
    coord_flip() + ggtitle("Histogram of price") +
    ylab("Count") +
    xlab("price") + 
    theme(plot.title = element_text(hjust = 0.5),legend.position=c(0.9,0.8), legend.background = element_rect(fill="grey90",
                                                                                                              size=0.5, linetype="solid", 
                                                                                                              colour ="black"))
# Change plot size to 5 x 4
options(repr.plot.width=9, repr.plot.height=6)

p1 <- ggplot(data_set, aes(x=bedrooms, y=price)) + 
    geom_point(shape=1) +  
    geom_smooth(method=lm , color="blue", se=FALSE)+
    ggtitle("bedrooms") +
    theme(plot.title = element_text(hjust = 0.4))

p2 <- ggplot(data_set, aes(x=rooms, y=price)) + 
    geom_point(shape=1) +  
    geom_smooth(method=lm , color="blue", se=FALSE)+
    ggtitle("rooms") +
    theme(plot.title = element_text(hjust = 0.4))

p3 <- ggplot(data_set, aes(x=tiempo, y=price)) + 
    geom_point(shape=1) +  
    geom_smooth(method=lm , color="blue", se=FALSE)+
    ggtitle("tiempo") +
    theme(plot.title = element_text(hjust = 0.4))

p4 <- ggplot(data_set, aes(x=latlon2, y=price)) + 
    geom_point(shape=1) +  
    geom_smooth(method=lm , color="blue", se=FALSE)+
    ggtitle("latlon2") +
    theme(plot.title = element_text(hjust = 0.4))

library(gridExtra)
grid.arrange(p1, p2,p3,p4)

## Hago one-hot-encoding

one_hot_training_set <- one_hot_sparse(data_set %>% select(-title, -description, -id, -ad_type))
one_hot_training_set <- cbind(one_hot_training_set, X_bow)

## Separo el conjunto de evaluación

training_set <- one_hot_training_set[data_set$created_on < "2019-08-01",]
evaluation_set <- one_hot_training_set[data_set$created_on >= "2019-08-01",]

## Separo un conjunto de validación (10% de las observaciones, podría hacerse algo mejor)

valid_index <- c(4000:5000)  
train_set <- training_set[setdiff(1:nrow(training_set), valid_index),]
valid_set <- training_set[valid_index,]

## Entreno un modelo que xgboost

dtrain <- xgb.DMatrix(data = train_set[,setdiff(colnames(train_set), "price")],
                      label = train_set[,"price"])

dvalid <- xgb.DMatrix(data = valid_set[,setdiff(colnames(valid_set), "price")],
                      label = valid_set[,"price"])

base_model <- xgb.train(data = dtrain,
                        params = list(max_depth = 35,
                                      eta = 0.1, 
                                      gamma = 0.1,
                                      colsample_bytree = 0.9,
                                      subsample = 0.9,
                                      min_child_weight = 0),
                        nrounds = 100,
                        watchlist = list(train = dtrain, valid = dvalid),
                        objective = "reg:squarederror",
                        eval.metric = "rmse",
                        print_every_n = 2)



asd <- xgb.importance(model=base_model)

VARS_TO_KEEP <- asd[seq(from = 1 , to = 8000), ]$Feature #El 500 indica cuantas variables toma mi modelo.
one_hot_training_set <- one_hot_training_set[, c(VARS_TO_KEEP, "price")]

## Genero las predicciones sobre el conjunto de evaluación

deval <- xgb.DMatrix(data = evaluation_set[,setdiff(colnames(evaluation_set), "price")])
predicciones <- exp(predict(base_model, newdata=deval))-1

## Guardo las prediccines en un archivo para subir en Kaggle

predicciones <- data.frame(id=data_set[data_set$created_on >= "2019-08-01", "id"],
                           price=predicciones)

write.table(predicciones, "predicciones_base.txt", sep=",",
            row.names=FALSE, quote=FALSE)

