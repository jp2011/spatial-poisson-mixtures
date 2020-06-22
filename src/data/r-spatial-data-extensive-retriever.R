#!/usr/local/bin/Rscript

# Set this to the repository root
setwd("./")

libraries <- c("rgdal",
               "rgeos",
               "nomisr",
               "raster",
               "lubridate",
               "optparse",
               "dplyr",
               "DBI",
               "xtable",
               "spdep",
               "flexmix",
               "MASS",
               "GGally",
               "AER"
)
lapply(libraries, library, character.only = TRUE)
source(file.path("..", "data", "r-func-utils.R"))
source(file.path("..", "data", "r-func-interpolation.R"))


# Household Density
###############################################################################
#                              Household counts                               #
###############################################################################

Get.Household.Count <- function(comp.grid, measurement.date, soc.econ.folder,
                                geom.name="lsoa") {

  geom.measured <- Get.Geometry.By.Name(geom.name)
  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid, 
                                  measurement.spdf = geom.measured, 
                                  covariate.names = c("HHOLDS"), extensive = T)

  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))

  cov.names <- "household.count"
  cov.values <- (output.spdf@data)["HHOLDS"]
  names(cov.values) <- cov.names
  centroids.with.data <- data.frame(centroids.cords, cov.values)
  return(centroids.with.data)
}


###############################################################################
#                              Uneployment data                               #
###############################################################################
Get.Unemployment.Data <- function(comp.grid, 
                                  measurement.date,  
                                  soc.econ.folder,
                                  geom.name = "lsoa") {

  geom.measured <- Get.Geometry.By.Name(geom.name)

  current.year <- year(measurement.date)
  csv.file.name <- file.path(soc.econ.folder, 
                             paste("nomis-unemployed-count-", current.year, "-",
                                   toupper(geom.name), ".csv", sep=""))

  # check if the CSV exists, if yes load from there else retrieve and save
  if (file.exists(csv.file.name)) {
    unemployment.data <- read.csv(csv.file.name)
  } else {
    current.id <- "NM_556_1"
    unemployment.data <- nomis_get_data(id = current.id, 
                                        geography = Get.Geo.Nomis.Code(geom.name),
                                        date = c(current.year),
                                        rural_urban = 0,
                                        measures=c("20100", "20301"),
                                        cell=c(8),
                                        select=c("GEOGRAPHY_CODE", "OBS_VALUE", "MEASURES_NAME"))
    unemployment.data <- tidyr::spread(unemployment.data, MEASURES_NAME, OBS_VALUE) %>%
      dplyr::rename(unemployment.count = Value, unemployment.percent = Percent) %>%
      dplyr::mutate(unemployment.percent = unemployment.percent / 100)
    write.csv(unemployment.data, csv.file.name, row.names=F)
  }
  
  # merge the data with the map
  geom.measured@data <- sp::merge(geom.measured@data,
                                  unemployment.data,
                                  by.x = Get.Geo.Id.Column.Name(geom.name),
                                  by.y = "GEOGRAPHY_CODE")

  count.data.spdf <- Weighted.Overlay(desired.spdf = comp.grid, 
                                  measurement.spdf = geom.measured, 
                                  covariate.names = c("unemployment.count"),
                                  extensive = T)

  percent.data.spdf <- Weighted.Overlay(desired.spdf = comp.grid, 
                                        measurement.spdf = geom.measured, 
                                        covariate.names = c("unemployment.percent"),
                                        extensive = F)
  
  count.data.cov.values <- (count.data.spdf@data)[c('unemployment.count')]
  percent.data.cov.values <- (percent.data.spdf@data)[c('unemployment.percent')]

  centroids.with.data <- data.frame(coordinates(gCentroid(count.data.spdf, byid=TRUE)), 
                                    count.data.cov.values,
                                    percent.data.cov.values)
  return(centroids.with.data)
}


###############################################################################
#                            Eating/Drinking POIs                             #
###############################################################################
Get.Poi.Eating.Drinking.Data <- function(comp.grid.spdf) {
  poi.data.fname <- file.path(getwd(), "../../data", "external", "EDINA",
                              "Download_poi-data_1210110", "poi_2911358",
                              "poi.csv")
  poi.data <- read.csv(poi.data.fname, header = TRUE, sep = "|")

  selected.pois <- poi.data %>%
    filter(PointX.Classification.Code >= 01020000 & PointX.Classification.Code < 01030000) %>%
    mutate(COUNT=1)

  selected.pois.points <- SpatialPointsDataFrame(coords = selected.pois[c("Feature.Easting", "Feature.Northing")], 
                                                 data = selected.pois, 
                                                 proj4string = CRS(BNG.CRS.PROJECTION.CODE))
  # project it to the target geometry
  selected.pois.points <- sp::spTransform(selected.pois.points, 
                                          CRS(proj4string(comp.grid.spdf))) 
  proj4string(selected.pois.points) <- proj4string(comp.grid.spdf)
  selected.pois.points <- selected.pois.points[comp.grid.spdf, ] # clip it

  selected.pois.count <- aggregate(selected.pois.points[, "COUNT"], comp.grid.spdf, sum)
  selected.pois.count@data$COUNT[is.na(selected.pois.count@data$COUNT)] <-0

  slot.centroids <- gCentroid(selected.pois.count, byid=TRUE)@coords
  slot.values <- selected.pois.count@data

  centroids.with.counds.df <- data.frame(slot.centroids, count=slot.values[, "COUNT"])

  column.name <- "poi.eatdrink.count"
  names(centroids.with.counds.df) <- c("x", "y", column.name)

  return(centroids.with.counds.df)
}


###############################################################################
#                              Sport facilities                               #
###############################################################################
Get.Sport.Entertainment.Pois <- function(comp.grid.spdf) {
  poi.data.fname <- file.path(getwd(), "../../data", "external", "EDINA",
                              "Download_poi-data_1210110", "poi_2911358",
                              "poi.csv")
  poi.data <- read.csv(poi.data.fname, header = TRUE, sep = "|")

  selected.pois <- poi.data %>%
    filter(PointX.Classification.Code >= 04210000 & PointX.Classification.Code < 04260000) %>%
    mutate(COUNT=1)

  selected.pois.points <- SpatialPointsDataFrame(coords = selected.pois[c("Feature.Easting", "Feature.Northing")], 
                                                 data = selected.pois, 
                                                 proj4string = CRS(BNG.CRS.PROJECTION.CODE))
  # project it to the target geometry
  selected.pois.points <- sp::spTransform(selected.pois.points, 
                                          CRS(proj4string(comp.grid.spdf))) 
  proj4string(selected.pois.points) <- proj4string(comp.grid.spdf)
  selected.pois.points <- selected.pois.points[comp.grid.spdf, ] # clip it

  selected.pois.count <- aggregate(selected.pois.points[, "COUNT"], comp.grid.spdf, sum)
  selected.pois.count@data$COUNT[is.na(selected.pois.count@data$COUNT)] <-0

  slot.centroids <- gCentroid(selected.pois.count, byid=TRUE)@coords
  slot.values <- selected.pois.count@data

  centroids.with.counds.df <- data.frame(slot.centroids, count=slot.values[, "COUNT"])

  column.name <- "poi.sport.entertainment.count"
  names(centroids.with.counds.df) <- c("x", "y", column.name)

  return(centroids.with.counds.df)
}

###############################################################################
#                         Accommodation facilities 
###############################################################################
Get.Accommodation.Pois <- function(comp.grid.spdf) {
  poi.data.fname <- file.path(getwd(), "../../data", "external", "EDINA",
                              "Download_poi-data_1210110", "poi_2911358",
                              "poi.csv")
  poi.data <- read.csv(poi.data.fname, header = TRUE, sep = "|")

  selected.pois <- poi.data %>%
    filter(PointX.Classification.Code >= 01010000 & PointX.Classification.Code < 01020000) %>%
    mutate(COUNT=1)

  selected.pois.points <- SpatialPointsDataFrame(coords = selected.pois[c("Feature.Easting", "Feature.Northing")], 
                                                 data = selected.pois, 
                                                 proj4string = CRS(BNG.CRS.PROJECTION.CODE))
  # project it to the target geometry
  selected.pois.points <- sp::spTransform(selected.pois.points, 
                                          CRS(proj4string(comp.grid.spdf))) 
  proj4string(selected.pois.points) <- proj4string(comp.grid.spdf)
  selected.pois.points <- selected.pois.points[comp.grid.spdf, ] # clip it

  selected.pois.count <- aggregate(selected.pois.points[, "COUNT"], comp.grid.spdf, sum)
  selected.pois.count@data$COUNT[is.na(selected.pois.count@data$COUNT)] <-0

  slot.centroids <- gCentroid(selected.pois.count, byid=TRUE)@coords
  slot.values <- selected.pois.count@data

  centroids.with.counds.df <- data.frame(slot.centroids, count=slot.values[, "COUNT"])

  column.name <- "poi.accommodation.count"
  names(centroids.with.counds.df) <- c("x", "y", column.name)

  return(centroids.with.counds.df)
}


###############################################################################
#                            Education and health                             #
###############################################################################
Get.Education.Health.Pois <- function(comp.grid.spdf) {
  poi.data.fname <- file.path(getwd(), "../../data", "external", "EDINA",
                              "Download_poi-data_1210110", "poi_2911358",
                              "poi.csv")
  poi.data <- read.csv(poi.data.fname, header = TRUE, sep = "|")

  selected.pois <- poi.data %>%
    filter(PointX.Classification.Code >= 05210000 & PointX.Classification.Code < 05330000) %>%
    mutate(COUNT=1)

  selected.pois.points <- SpatialPointsDataFrame(coords = selected.pois[c("Feature.Easting", "Feature.Northing")], 
                                                 data = selected.pois, 
                                                 proj4string = CRS(BNG.CRS.PROJECTION.CODE))
  # project it to the target geometry
  selected.pois.points <- sp::spTransform(selected.pois.points, 
                                          CRS(proj4string(comp.grid.spdf))) 
  proj4string(selected.pois.points) <- proj4string(comp.grid.spdf)
  selected.pois.points <- selected.pois.points[comp.grid.spdf, ] # clip it

  selected.pois.count <- aggregate(selected.pois.points[, "COUNT"], comp.grid.spdf, sum)
  selected.pois.count@data$COUNT[is.na(selected.pois.count@data$COUNT)] <-0

  slot.centroids <- gCentroid(selected.pois.count, byid=TRUE)@coords
  slot.values <- selected.pois.count@data

  centroids.with.counds.df <- data.frame(slot.centroids, count=slot.values[, "COUNT"])

  column.name <- "poi.edu.health.count"
  names(centroids.with.counds.df) <- c("x", "y", column.name)

  return(centroids.with.counds.df)
}



###############################################################################
#                                 Retail POIs                                 #
###############################################################################
Get.Poi.Retail.And.Services.Data <- function(comp.grid.spdf) {
  poi.data.fname <- file.path(getwd(), '../../data', 'external', 'EDINA', "Download_poi-data_1210110", "poi_2911358", "poi.csv")
  poi.data <- read.csv(poi.data.fname, header = TRUE, sep = "|")

  selected.pois <- poi.data %>%
    filter((PointX.Classification.Code >= 09460000 & PointX.Classification.Code < 09500000) | (PointX.Classification.Code >= 02130000 & PointX.Classification.Code < 02140000)) %>%
    mutate(COUNT=1)

  selected.pois.points <- SpatialPointsDataFrame(coords = selected.pois[c("Feature.Easting", "Feature.Northing")], 
                                                 data = selected.pois, proj4string = CRS(BNG.CRS.PROJECTION.CODE))
  selected.pois.points <- sp::spTransform(selected.pois.points, CRS(proj4string(comp.grid.spdf))) # project it to the target geometry
  proj4string(selected.pois.points) <- proj4string(comp.grid.spdf)
  selected.pois.points <- selected.pois.points[comp.grid.spdf, ]

  selected.pois.count <- aggregate(selected.pois.points[, 'COUNT'], comp.grid.spdf, sum)
  selected.pois.count@data$COUNT[is.na(selected.pois.count@data$COUNT)] <- 0 

  slot.centroids <- gCentroid(selected.pois.count, byid=TRUE)@coords
  slot.values <- selected.pois.count@data

  centroids.with.counds.df <- data.frame(slot.centroids, count=slot.values[, "COUNT"])

  column.name <- 'poi.retail.count'
  names(centroids.with.counds.df) <- c("x", "y", column.name)

  return(centroids.with.counds.df)
}



###############################################################################
#                            Ethnic Heterogeneity                             #
###############################################################################
Get.Ethnic.Heterogeneity <- function(comp.grid, measurement.date, soc.econ.folder,
                                     geom.name="msoa") {

  geom.measured <- Get.Geometry.By.Name(geom.name)

  current.year <- year(measurement.date)
  csv.file.name <- file.path(soc.econ.folder, paste("nomis-ethnic-heterogeneity-", current.year, "-", toupper(geom.name), ".csv", sep=""))

  # check if the CSV exists, if yes load from there else retrieve and save
  if (file.exists(csv.file.name)) {
    ethnic.heterogeneity <- read.csv(csv.file.name)
  } else {
    current.id <- "NM_608_1"
    ethnicity.data <- nomis_get_data(id=current.id, 
                                     geography=Get.Geo.Nomis.Code(geom.name), 
                                     MEASURES="20301", 
                                     RURAL_URBAN="0",
                                     date="2011",
                                     select=c("GEOGRAPHY_CODE", "CELL", "CELL_NAME", "OBS_VALUE"),
                                     cell=c(100, 200, 300, 400, 500))
    ethnicity.data$OBS_VALUE <- as.numeric(ethnicity.data$OBS_VALUE)
    ethnicity.data$CELL <- as.numeric(as.character(ethnicity.data$CELL))

    ethnicity.data.white <- ethnicity.data %>%
      dplyr::filter(CELL==100) %>%
      dplyr::select(c("GEOGRAPHY_CODE", "OBS_VALUE"))

    ethnicity.data.mixed <- ethnicity.data %>%
      dplyr::filter(CELL==200) %>%
      dplyr::select(c("GEOGRAPHY_CODE", "OBS_VALUE"))

    ethnicity.data.asian <- ethnicity.data %>%
      dplyr::filter(CELL==300) %>%
      dplyr::select(c("GEOGRAPHY_CODE", "OBS_VALUE"))

    ethnicity.data.black <- ethnicity.data %>%
      dplyr::filter(CELL==400) %>%
      dplyr::select(c("GEOGRAPHY_CODE", "OBS_VALUE"))

    ethnicity.data.other <- ethnicity.data %>%
      dplyr::filter(CELL==500) %>%
      dplyr::select(c("GEOGRAPHY_CODE", "OBS_VALUE"))

    ethnicities <- cbind(ethnicity.data.white$OBS_VALUE,
                         ethnicity.data.mixed$OBS_VALUE,
                         ethnicity.data.asian$OBS_VALUE,
                         ethnicity.data.black$OBS_VALUE,
                         ethnicity.data.other$OBS_VALUE)

    heterogeneity.index <- 1 - rowSums(((ethnicities / 100) ** 2))

    ethnic.heterogeneity <- data.frame(GEOGRAPHY_CODE=ethnicity.data.white$GEOGRAPHY_CODE,
                                       OBS_VALUE=heterogeneity.index)

    write.csv(ethnic.heterogeneity, csv.file.name, row.names=F)
  }
  ethnic.heterogeneity.merged <- sp::merge(geom.measured@data, 
                                           ethnic.heterogeneity, 
                                           by.x = Get.Geo.Id.Column.Name(geom.name), 
                                           by.y = "GEOGRAPHY_CODE") %>%
    rename(ethnic.heterogeneity = OBS_VALUE) %>%
    mutate(logit.ethnic.heterogeneity = log(ethnic.heterogeneity) - log(1 - ethnic.heterogeneity))

  geom.measured@data <- ethnic.heterogeneity.merged

  cov.names <- c("ethnic.heterogeneity", "logit.ethnic.heterogeneity")
  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = cov.names,
                                  extensive = F)

  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))

  cov.values <- (output.spdf@data)[cov.names]
  names(cov.values) <- cov.names
  centroids.with.data <- data.frame(centroids.cords, cov.values)

  return(centroids.with.data)
}


###############################################################################
#                           Transport accessibility                           #
###############################################################################
Get.Transport.Accessibility <- function(comp.grid, measurement.date, soc.econ.folder) {
  geom.name <- "lsoa"
  geom.measured <- Get.Geometry.By.Name(geom.name)

  # This data is from 2015!
  csv.file.name <- file.path(soc.econ.folder, "tfl-ptal-accessibility.csv")
  accessibility.data <- read.csv(csv.file.name, sep=",", header=T)
  accessibility.data <- accessibility.data %>%
    dplyr::select(LSOA2011, AvPTAI2015) %>%
    dplyr::rename(accessibility = AvPTAI2015)

  # Merge with the LSOAs shapes
  accessibility.data.merged <- sp::merge(geom.measured@data, accessibility.data, by.x=Get.Geo.Id.Column.Name(geom.name), by.y="LSOA2011")

  geom.measured@data <- accessibility.data.merged

  acccess.col.names <- c("accessibility")
  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = acccess.col.names,
                                  extensive = F)
  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))

  cov.names <- acccess.col.names
  cov.values <- (output.spdf@data)[acccess.col.names]
  names(cov.values) <- cov.names
  centroids.with.data <- data.frame(centroids.cords, cov.values)
  return(centroids.with.data)
  
}


###############################################################################
#                             Population turnover                             #
###############################################################################
Get.Population.Turnover <- function(comp.grid, measurement.date, soc.econ.folder,
                                    geom.name="msoa") {

  geom.measured <- Get.Geometry.By.Name(geom.name)

  current.year <- year(measurement.date)
  csv.file.name <- file.path(soc.econ.folder, paste("nomis-population-moves-", current.year, "-", toupper(geom.name), ".csv", sep=""))

  # check if the CSV exists, if yes load from there else retrieve and save
  if (file.exists(csv.file.name)) {
    population.move <- read.csv(csv.file.name)
  } else {
    current.id <- "NM_1287_1"
    migration.data <- nomis_get_data(id = current.id, 
                                     geography = Get.Geo.Nomis.Code(geom.name), 
                                     MEASURES = "20100", 
                                     date = current.year,
                                     c_migr = c(0, 4, 5, 6),
                                     select=c("GEOGRAPHY_CODE","C_MIGR",   "OBS_VALUE")
                                     )

    population.move <- migration.data %>%
      tidyr::spread(C_MIGR, OBS_VALUE) %>%
      dplyr::rename(people.moved.in.overseas = `5`) %>%
      dplyr::rename(people.moved.in.uk = `4`) %>%
      dplyr::mutate(people.moved.in = people.moved.in.overseas + people.moved.in.uk) %>%
      dplyr::rename(people.moved.out = `6`) %>%
      dplyr::rename(people.current = `0`)

    write.csv(population.move, csv.file.name, row.names=F)
  }

  population.move.merged <- sp::merge(geom.measured@data, population.move, 
                                      by.x = Get.Geo.Id.Column.Name(geom.name),
                                      by.y = "GEOGRAPHY_CODE")
  geom.measured@data <- population.move.merged

  cov.names <- c("people.moved.in", "people.moved.out", "people.current")
  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = cov.names,
                                  extensive = T)

  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))

  cov.values <- (output.spdf@data)[cov.names]
  names(cov.values) <- cov.names
  centroids.with.data <- data.frame(centroids.cords, cov.values)
  return(centroids.with.data)
}


###############################################################################
#                                Dwelling type                                #
###############################################################################
Get.Dwelling.Type.Data <- function(comp.grid, measurement.date,
                                     soc.econ.folder, geom.name="lsoa") {

  geom.measured <- Get.Geometry.By.Name(geom.name)
  geo.id.col.name <- Get.Geo.Id.Column.Name(geom.name)
  col.names <- c("all.houses", "detached.houses", "semidetached.houses", "terraced.houses")
  col.names.percent <- as.vector(sapply(col.names, function(x) paste(x, '.fraction', sep="")))
  col.names.count <- as.vector(sapply(col.names, function(x) paste(x, '.count', sep="")))

  current.year <- year(measurement.date)
  csv.file.name <- file.path(soc.econ.folder, paste("nomis-dwelling-types-", current.year, "-", toupper(geom.name), ".csv", sep=""))

  # check if the CSV exists, if yes load from there else retrieve and save
  if (file.exists(csv.file.name)) {
    house.types.all <- read.csv(csv.file.name)
  } else {
    current.id <- "NM_533_1"
    all.data <- nomis_get_data(id = current.id, geography = Get.Geo.Nomis.Code(geom.name), 
                               MEASURES=c("20100", "20301"),
                               rural_urban=0)

    house.types.percents <- all.data %>%
      dplyr::filter(MEASURES == '20301') %>%
      dplyr::filter(DWELLING_TYPE %in% c("2", "3", "4", "5")) %>%
      dplyr::mutate(OBS_VALUE = OBS_VALUE / 100) %>%
      dplyr::select(GEOGRAPHY_CODE, DWELLING_TYPE, OBS_VALUE) %>%
      tidyr::spread(DWELLING_TYPE, OBS_VALUE)
    names(house.types.percents) <- c("GEOGRAPHY_CODE", col.names.percent)

     house.types.counts <- all.data %>%
      dplyr::filter(MEASURES == '20100') %>%
      dplyr::filter(DWELLING_TYPE %in% c("2", "3", "4", "5")) %>%
      dplyr::select(GEOGRAPHY_CODE, DWELLING_TYPE, OBS_VALUE) %>%
      tidyr::spread(DWELLING_TYPE, OBS_VALUE)
    names(house.types.counts) <- c("GEOGRAPHY_CODE", col.names.count)

    house.types.all <- house.types.counts %>%
      dplyr::full_join(house.types.percents, by='GEOGRAPHY_CODE')

    write.csv(house.types.all, csv.file.name, row.names=F)
  }

  # Merge with the map
  house.types.all.merged <- sp::merge(geom.measured@data, house.types.all,
                                         by.x = geo.id.col.name,
                                         by.y = "GEOGRAPHY_CODE")

  geom.measured@data <- house.types.all.merged

  percent.output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = col.names.percent,
                                  extensive = F)

  count.output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = col.names.count,
                                  extensive = T)

  centroids.cords <- coordinates(gCentroid(percent.output.spdf, byid=TRUE))

  count.cov.values <- (count.output.spdf@data)[col.names.count]
  names(count.cov.values) <- col.names.count

  percent.cov.values <- (percent.output.spdf@data)[col.names.percent]
  names(percent.cov.values) <- col.names.percent


  centroids.with.data <- data.frame(centroids.cords, count.cov.values, percent.cov.values)
  return(centroids.with.data)
}



###############################################################################
#                            Occupation variation                             #
###############################################################################
Get.Skill.Occupation <- function(comp.grid, measurement.date, soc.econ.folder,
                                 geom.name='lsoa') {

  geom.measured <- Get.Geometry.By.Name(geom.name)

  current.year <- year(measurement.date)
  csv.file.name <- file.path(soc.econ.folder, paste("nomis-occupation-variation-", current.year, "-", toupper(geom.name), ".csv", sep=""))

  # check if the CSV exists, if yes load from there else retrieve and save
  if (file.exists(csv.file.name)) {
    occupation.heterogeneity <- read.csv(csv.file.name)
  } else {
    current.id <- "NM_1518_1"
    occupation.data <- nomis_get_data(id=current.id, 
                                      geography=Get.Geo.Nomis.Code(geom.name), 
                                      MEASURES=c("20301"), 
                                      date=current.year,
                                      CELL=c(1,2,3,4,5,6,7,8,9),
                                      select=c("GEOGRAPHY_CODE", "CELL", "CELL_NAME", "OBS_VALUE"))
    occupation.data$OBS_VALUE <- as.numeric(occupation.data$OBS_VALUE)
    occupation.data$CELL <- as.numeric(as.character(occupation.data$CELL))

    occupation.heterogeneity <- occupation.data %>%
      filter(CELL > 0) %>%                          # ignore the aggregate
      mutate(OBS_VALUE = OBS_VALUE / 100) %>%
      group_by(GEOGRAPHY_CODE) %>%
      summarise(occupation.variation = 1 - sum(OBS_VALUE ** 2))

    write.csv(occupation.heterogeneity, csv.file.name, row.names=F)
  }

  # Merge with the LSOAs shapes
  occupation.skill.merged <- sp::merge(geom.measured@data, occupation.heterogeneity,
                                       by.x = Get.Geo.Id.Column.Name(geom.name),
                                       by.y="GEOGRAPHY_CODE")
  geom.measured@data <- occupation.skill.merged

  cov.names <- c("occupation.variation")
  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = cov.names,
                                  extensive = F)

  cov.values <- (output.spdf@data)[cov.names]
  names(cov.values) <- cov.names
  cov.values <- cov.values %>%
    dplyr::mutate(logit.occupation.variation = log(occupation.variation) - log(1 - occupation.variation))
  
  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))
  centroids.with.data <- data.frame(centroids.cords, cov.values)
  return(centroids.with.data)
}


###############################################################################
#                               Housing Tenure                                #
###############################################################################
Get.House.Tenure <- function(comp.grid, measurement.date, soc.econ.folder,
                             geom.name='lsoa') {
  
  col.names <- c("tenure.owned", "tenure.rented.social", "tenure.rented.private", "tenure.other")
  col.names.percent <- as.vector(sapply(col.names, function(x) paste(x, '.fraction', sep="")))
  col.names.count <- as.vector(sapply(col.names, function(x) paste(x, '.count', sep="")))
  
  geom.measured <- Get.Geometry.By.Name(geom.name)
  current.year <- year(measurement.date)
  csv.file.name <- file.path(soc.econ.folder, paste("nomis-house-counts-", current.year, "-", toupper(geom.name), ".csv", sep=""))

  # check if the CSV exists, if yes load from there else retrieve and save
  if (file.exists(csv.file.name)) {
    house.tenure <- read.csv(csv.file.name)
  } else {
    current.id <- "NM_619_1"
    house.tenure.data <- nomis_get_data(id=current.id, 
                                        geography=Get.Geo.Nomis.Code(geom.name), 
                                        RURAL_URBAN=0,
                                        MEASURES=c("20100", "20301"), 
                                        date=current.year,
                                        cell=c(100, 200, 300, 8, 3), # owned, social rented, private rented, rent free, shart ownership 
                                        select=c("GEOGRAPHY_CODE", "MEASURES", "CELL", "CELL_NAME", "OBS_VALUE"))



    tenure.owned <- house.tenure.data %>%
      dplyr::filter(CELL == 100) %>%
      dplyr::group_by(GEOGRAPHY_CODE, MEASURES) %>%
      dplyr::summarise(OBS_VALUE=sum(OBS_VALUE)) %>%
      dplyr::select(GEOGRAPHY_CODE, MEASURES, OBS_VALUE) %>%
      tidyr::spread(MEASURES, OBS_VALUE) %>%
      dplyr::rename(tenure.owned.fraction = `20301`) %>%
      dplyr::mutate(tenure.owned.fraction = tenure.owned.fraction / 100) %>%
      dplyr::rename(tenure.owned.count = `20100`)

    tenure.rented.social <- house.tenure.data %>%
      dplyr::filter(CELL == 200) %>%
      dplyr::group_by(GEOGRAPHY_CODE, MEASURES) %>%
      dplyr::summarise(OBS_VALUE=sum(OBS_VALUE)) %>%
      dplyr::select(GEOGRAPHY_CODE, MEASURES, OBS_VALUE) %>%
      tidyr::spread(MEASURES, OBS_VALUE) %>%
      dplyr::rename(tenure.rented.social.fraction = `20301`) %>%
      dplyr::mutate(tenure.rented.social.fraction = tenure.rented.social.fraction / 100) %>%
      dplyr::rename(tenure.rented.social.count = `20100`)

    tenure.rented.private <- house.tenure.data %>%
      dplyr::filter(CELL == 300) %>%
      dplyr::group_by(GEOGRAPHY_CODE, MEASURES) %>%
      dplyr::summarise(OBS_VALUE=sum(OBS_VALUE)) %>%
      dplyr::select(GEOGRAPHY_CODE, MEASURES, OBS_VALUE) %>%
      tidyr::spread(MEASURES, OBS_VALUE) %>%
      dplyr::rename(tenure.rented.private.fraction = `20301`) %>%
      dplyr::mutate(tenure.rented.private.fraction = tenure.rented.private.fraction / 100) %>%
      dplyr::rename(tenure.rented.private.count = `20100`)

    tenure.other <- house.tenure.data %>%
      dplyr::filter(CELL == 3 | CELL == 8) %>%
      dplyr::group_by(GEOGRAPHY_CODE, MEASURES) %>%
      dplyr::summarise(OBS_VALUE=sum(OBS_VALUE)) %>%
      dplyr::select(GEOGRAPHY_CODE, MEASURES, OBS_VALUE) %>%
      tidyr::spread(MEASURES, OBS_VALUE) %>%
      dplyr::rename(tenure.other.fraction = `20301`) %>%
      dplyr::mutate(tenure.other.fraction = tenure.other.fraction / 100) %>%
      dplyr::rename(tenure.other.count = `20100`)

    house.tenure <- tenure.owned %>%
      dplyr::inner_join(tenure.rented.social, by = 'GEOGRAPHY_CODE') %>%
      dplyr::inner_join(tenure.rented.private, by = 'GEOGRAPHY_CODE') %>%
      dplyr::inner_join(tenure.other, by = 'GEOGRAPHY_CODE')

    write.csv(house.tenure, csv.file.name, row.names=F)
  }
  house.tenure.merged <- sp::merge(geom.measured@data, house.tenure, by.x=Get.Geo.Id.Column.Name(geom.name), by.y="GEOGRAPHY_CODE")
  geom.measured@data <- house.tenure.merged

  percent.output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                          measurement.spdf = geom.measured,
                                          covariate.names = col.names.percent,
                                          extensive = F)

  count.output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                        measurement.spdf = geom.measured,
                                        covariate.names = col.names.count,
                                        extensive = T)

  centroids.cords <- coordinates(gCentroid(percent.output.spdf, byid=TRUE))

  count.cov.values <- (count.output.spdf@data)[col.names.count]
  names(count.cov.values) <- col.names.count

  percent.cov.values <- (percent.output.spdf@data)[col.names.percent]
  names(percent.cov.values) <- col.names.percent


  centroids.with.data <- data.frame(centroids.cords, count.cov.values, percent.cov.values)
  return(centroids.with.data)
}


###############################################################################
#                            Household composition                            #
###############################################################################
Get.Household.Types <- function(comp.grid, measurement.date, soc.econ.folder,
                                      geom.name='lsoa') {

  col.names <- c("single.parent.household", "one.person.household", "couple.with.children")
  col.names.percent <- as.vector(sapply(col.names, function(x) paste(x, '.fraction', sep="")))
  col.names.count <- as.vector(sapply(col.names, function(x) paste(x, '.count', sep="")))

  geom.measured <- Get.Geometry.By.Name(geom.name)
  current.year <- year(measurement.date)
  csv.file.name <- file.path(soc.econ.folder,
                             paste("nomis-household-type-",
                                   current.year, "-", toupper(geom.name),
                                   ".csv", sep=""))

  if (file.exists(csv.file.name)) {
    household.composition <- read.csv(csv.file.name)
  } else {
    current.id <- "NM_516_1"
    household.composition.raw <- nomis_get_data(id=current.id, 
                                      geography=Get.Geo.Nomis.Code(geom.name), 
                                      RURAL_URBAN=0,
                                      MEASURES=c("20100", "20301"),
                                      date=current.year)

    # percentage data
    household.composition.percent <- household.composition.raw %>%
      dplyr::filter(MEASURES == '20301') %>%
      dplyr::mutate(OBS_VALUE = OBS_VALUE / 100) %>%
      dplyr::rename(HH_TYPE_CODE = C_AHTHUK11_SORTORDER) %>%
      dplyr::filter(HH_TYPE_CODE > 0) %>%
      dplyr::select(GEOGRAPHY_CODE, HH_TYPE_CODE, OBS_VALUE) %>%
      tidyr::spread(HH_TYPE_CODE, OBS_VALUE) %>%
      dplyr::rename_all(make.names) %>%
      dplyr::mutate(single.parent.household.fraction = X8 + X9) %>%
      dplyr::mutate(one.person.household.fraction = X1) %>%
      dplyr::mutate(couple.with.children.fraction = X2 + X4 + X6) %>%
      dplyr::select(GEOGRAPHY_CODE, single.parent.household.fraction, one.person.household.fraction, couple.with.children.fraction)
      
    # count data
    household.composition.count <- household.composition.raw %>%
      dplyr::filter(MEASURES == '20100') %>%
      dplyr::rename(HH_TYPE_CODE = C_AHTHUK11_SORTORDER) %>%
      dplyr::filter(HH_TYPE_CODE > 0) %>%
      dplyr::select(GEOGRAPHY_CODE, HH_TYPE_CODE, OBS_VALUE) %>%
      tidyr::spread(HH_TYPE_CODE, OBS_VALUE) %>%
      dplyr::rename_all(make.names) %>%
      dplyr::mutate(single.parent.household.count = X8 + X9) %>%
      dplyr::mutate(one.person.household.count = X1) %>%
      dplyr::mutate(couple.with.children.count = X2 + X4 + X6) %>%
      dplyr::select(GEOGRAPHY_CODE, single.parent.household.count, one.person.household.count, couple.with.children.count)

    household.composition <- household.composition.percent %>%
      dplyr::inner_join(household.composition.count, by = 'GEOGRAPHY_CODE')
    write.csv(household.composition, csv.file.name, row.names=F)
  }

  # Merge with the LSOAs shapes
  data.merged <- sp::merge(geom.measured@data, household.composition,
                           by.x = Get.Geo.Id.Column.Name(geom.name),
                           by.y = "GEOGRAPHY_CODE")
  geom.measured@data <- data.merged

  percent.output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                          measurement.spdf = geom.measured,
                                          covariate.names = col.names.percent,
                                          extensive = F)

  count.output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                        measurement.spdf = geom.measured,
                                        covariate.names = col.names.count,
                                        extensive = T)

  centroids.cords <- coordinates(gCentroid(percent.output.spdf, byid=TRUE))

  count.cov.values <- (count.output.spdf@data)[col.names.count]
  names(count.cov.values) <- col.names.count

  percent.cov.values <- (percent.output.spdf@data)[col.names.percent]
  names(percent.cov.values) <- col.names.percent

  centroids.with.data <- data.frame(centroids.cords, count.cov.values, percent.cov.values)
  return(centroids.with.data)
}


###############################################################################
#                            Mean household income                            #
###############################################################################
Get.Mean.HH.Income <- function (comp.grid, measurement.date, soc.econ.folder) {
  geom.name <- "lsoa"
  geom.measured <- Get.Geometry.By.Name(geom.name)

  hh.income.fname <- file.path(soc.econ.folder, 'gla-hh-income-lsoa-estimate.csv')
  hh.income.lsoa <- read.csv(file = hh.income.fname)

  # Merge with the LSOAs shapes
  data.merged <- sp::merge(geom.measured@data, hh.income.lsoa,
                           by.x = Get.Geo.Id.Column.Name(geom.name),
                           by.y = "LSOA11CD")
  geom.measured@data <- data.merged

  cov.names <- c("mean.hh.income")

  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = cov.names,
                                  extensive = F)

  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))

  cov.values <- (output.spdf@data)[cov.names]
  names(cov.values) <- cov.names
  centroids.with.data <- data.frame(centroids.cords, cov.values)
  return(centroids.with.data)
}


###############################################################################
#                                House Prices                                 #
###############################################################################
Get.House.Prices <- function (comp.grid, measurement.date, soc.econ.folder) {
  
  geom.name <- "lsoa"
  geom.measured <- Get.Geometry.By.Name(geom.name)

  house.prices.fname <- file.path(soc.econ.folder, 'gla-house-price-lsoa-estimate.csv')
  house.prices.lsoa <- read.csv(file = house.prices.fname)

  # Merge with the LSOAs shapes
  data.merged <- sp::merge(geom.measured@data, house.prices.lsoa,
                           by.x = Get.Geo.Id.Column.Name(geom.name),
                           by.y = "LSOA11CD")
  geom.measured@data <- data.merged

  cov.names <- c("house.price")
  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = cov.names,
                                  extensive = F)

  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))

  cov.values <- (output.spdf@data)[cov.names]
  names(cov.values) <- cov.names
  centroids.with.data <- data.frame(centroids.cords, cov.values)
  return(centroids.with.data)
}


###############################################################################
#                                  Age data                                   #
###############################################################################
Get.Age.Data <- function(comp.grid, measurement.date, soc.econ.folder,
                         geom.name='lsoa') {

  geom.measured <- Get.Geometry.By.Name(geom.name)

  current.year <- year(measurement.date)
  csv.file.name <- file.path(soc.econ.folder, paste("nomis-population-median-age-", current.year, "-", toupper(geom.name), ".csv", sep=""))
  if (file.exists(csv.file.name)) {
    median.age <- read.csv(csv.file.name)
  } else {
    current.id <- "NM_145_1"
   
    median.age <- nomis_get_data(id=current.id, geography = Get.Geo.Nomis.Code(geom.name), MEASURES="20100",
                                 cell=18, rural_urban=0, date=c(current.year), select=c("GEOGRAPHY_CODE","OBS_VALUE"))
    colnames(median.age) <- c("GEOGRAPHY_CODE", "median.age")
    write.csv(median.age, csv.file.name, row.names=F)
  }

  # Merge with the LSOAs shapes
  age.data.merged <- sp::merge(geom.measured@data, median.age, by.x=Get.Geo.Id.Column.Name(geom.name), by.y="GEOGRAPHY_CODE")
  geom.measured@data <- age.data.merged

  col.names <- c("median.age")
  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid,
                                  measurement.spdf = geom.measured,
                                  covariate.names = col.names,
                                  extensive = F)

  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))
  
  cov.names <- col.names
  cov.values <- (output.spdf@data)[col.names]
  names(cov.values) <- cov.names
  centroids.with.data <- data.frame(centroids.cords, cov.values)
  return(centroids.with.data)
}


###############################################################################
#                             Urbanisation Level                              #
###############################################################################
Get.Urbanisation.Level <- function(comp.grid) {

  # This data is from 2015!
  ldn.urban.grid.shp.fname <- file.path(getwd(), '../../data', 'interim', 'ldn-urban-area.shp')

  ldn.urban.data.spdf <- readOGR(dsn=ldn.urban.grid.shp.fname)
  ldn.urban.data.spdf <- sp::spTransform(ldn.urban.data.spdf, CRS(proj4string(comp.grid))) # project it to the target geometry

  output.spdf <- Weighted.Overlay(desired.spdf = comp.grid, measurement.spdf = ldn.urban.data.spdf,
                                  covariate.names = c('URBAN', 'URBAN_S'))
  centroids.cords <- coordinates(gCentroid(output.spdf, byid=TRUE))

  cov.names <- c('urban.proportion', 'urban.suburban.proportion') 
  cov.values <- (output.spdf@data)[c('URBAN', 'URBAN_S')]
  names(cov.values) <- cov.names
  centroids.with.data <- data.frame(centroids.cords, cov.values)
  return(centroids.with.data)
}


###############################################################################
###############################################################################
#                                  MAIN PART                                  #
###############################################################################
###############################################################################
default.start.date <-"2015-01-01"
default.end.date <- "2015-12-31"
default.resolution <- 400 
default.test.set <- 0.0


###############################################################################
#                                   PARSING                                   #
###############################################################################

option_list = list(
                   make_option(c("-r", "--resolution"), type="integer", default=default.resolution, 
                               help="Resolution for counting, [default= %default]", metavar="integer"),
                   make_option(c("-t", "--testset"), type="double", default=default.test.set, 
                               help="Size of the test set, [default= %default]", metavar="double"),
                   make_option(c("-s", "--startdate"), type="character", default=default.start.date, 
                               help="Start date.", metavar="character"),
                   make_option(c("-e", "--enddate"), type="character", default=default.end.date, 
                               help="End date.", metavar="character")
                   );

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

resolution <- opt$resolution
start.date <- ymd(opt$startdate)
end.date <- ymd(opt$enddate)
test.set <- opt$testset


###############################################################################
#                             SET MISC VARIABLES                              #
###############################################################################

LDN.GRID.SPDF <- Get.Ldn.Grid.Spdf(resolution)

sqlite_db_file_name <- file.path(getwd(), "../../data", "interim", "crime.db")
datafolder <- file.path(getwd(), "../../data")
soc.econ.folder <- file.path(datafolder, "raw", "socio-economic")
lsoa.shp.fname <- file.path(getwd(), "../../data", "external", "ESRI", "LSOA_2011_London_gen_MHW.shp")

collection.unit <- 'lsoa'
cov.method <- 'weighted'


###############################################################################
#                               CRIME RETRIEVAL                               #
###############################################################################
Retrieve.Crime.Data <- function(target.geom.spdf, start.date, end.date, sqlite_db_file_name, leave.out.percent=0.0) {

  con <- dbConnect(RSQLite::SQLite(), dbname=sqlite_db_file_name)
  query.str <- paste("SELECT * FROM LONDON WHERE MONTH >=\"", ym(start.date), "\" AND MONTH <=\"", ym(end.date), "\"", sep="")
  sql.query <- dbSendQuery(con, query.str)
  crime.data <- dbFetch(sql.query, n = -1)
  dbClearResult(sql.query)
  dbDisconnect(conn=con)

  if (nrow(crime.data) == 0) { return(data.frame()) } 

  crime.data <- crime.data %>%
    tidyr::drop_na() %>%
    dplyr::mutate(COUNT = 1) %>%
    dplyr::select(-c(DESCRIPTION))

  crime.type.tally <- crime.data %>% group_by(CRIME_TYPE) %>% tally()
  print(crime.type.tally)

  all.crimes.df <- data.frame()

  crime.types <- as.matrix(unique(crime.data['CRIME_TYPE']))
  for (crime.type in crime.types) {
    print(paste("Processing", crime.type, sep=" "))
    crime.type.data <- crime.data %>%
      filter(CRIME_TYPE == crime.type)

    # leave out percentage
    if (leave.out.percent > 0.0) {
      sample.size <- floor((1 - leave.out.percent) * nrow(crime.type.data))
      train.indices <- sample(seq_len(nrow(crime.type.data)), size = sample.size)
      crime.type.data <-  crime.type.data[train.indices, ]
    }
    # Create SPDF object and set the projections correctly
    Crime.Spatial<- SpatialPointsDataFrame(crime.type.data[c("LATITUDE","LONGITUDE")], crime.type.data, proj4string = CRS(GPS.CRS.PROJECTION.CODE))
    Crime.Spatial<- sp::spTransform(Crime.Spatial, CRS(proj4string(target.geom.spdf))) # project it to the target geometry
    proj4string(Crime.Spatial) <- proj4string(target.geom.spdf)
    Crime.Spatial <- Crime.Spatial[target.geom.spdf, ]  # Clip the data to the boundary

    crime.counts <- aggregate(Crime.Spatial[,"COUNT"], target.geom.spdf, sum)
    crime.counts@data$COUNT[is.na(crime.counts@data$COUNT)] <-0

    slot.centroids <- gCentroid(crime.counts, byid=TRUE)@coords
    slot.counts <- crime.counts@data

    centroids.with.counts.df <- data.frame(slot.centroids, slot.counts)
    names(centroids.with.counts.df) <- c('x', 'y', tolower(gsub('(\\ |-)+', '../..', crime.type)))

    if (length(all.crimes.df) == 0) {
      all.crimes.df <- centroids.with.counts.df 
    } else {
      all.crimes.df <- dplyr::full_join(all.crimes.df, centroids.with.counts.df, by = c("x", "y"))
    }
  }
  return(all.crimes.df)
}
all.crime <- Retrieve.Crime.Data(target.geom.spdf = LDN.GRID.SPDF, 
                                 start.date = start.date, end.date = end.date,
                                 sqlite_db_file_name = sqlite_db_file_name)

test.start.date <- end.date + 1 
test.end.date <- test.start.date + days(end.date - start.date)
test.set.crime <-  Retrieve.Crime.Data(target.geom.spdf = LDN.GRID.SPDF,
                                       start.date = test.start.date,
                                       end.date = test.end.date,
                                       sqlite_db_file_name = sqlite_db_file_name)
names(test.set.crime) <- gsub("^", "test.", names(test.set.crime))


###############################################################################
#                          MERGING THINGS TOGERTHER                           #
###############################################################################
merged.data <- dplyr::full_join(all.crime, test.set.crime, by = c("x" = "test.x",
                                                                  "y" = "test.y"))

household.count <- Get.Household.Count(LDN.GRID.SPDF, end.date, soc.econ.folder, "oa") %>%
  dplyr::mutate(log.household.count = log(1 + household.count))
merged.data <- dplyr::full_join(merged.data, household.count, by = c("x", "y"))

unemployment.data <- Get.Unemployment.Data(LDN.GRID.SPDF, end.date, soc.econ.folder, "oa") %>%
  dplyr::mutate(log.unemployment.count = log(1 + unemployment.count)) %>%
  dplyr::mutate(log.unemployment.percent = log(1 + unemployment.percent))
merged.data <- dplyr::full_join(merged.data, unemployment.data, by = c("x", "y"))

poi.retail <- Get.Poi.Retail.And.Services.Data(LDN.GRID.SPDF) %>%
  dplyr::mutate(log.poi.retail.count = log(1 + poi.retail.count))
merged.data <- dplyr::full_join(merged.data, poi.retail, by = c("x", "y"))

poi.eatdrink <- Get.Poi.Eating.Drinking.Data(LDN.GRID.SPDF) %>%
  dplyr::mutate(log.poi.eatdrink.count = log(1 + poi.eatdrink.count))
merged.data <- dplyr::full_join(merged.data, poi.eatdrink, by = c("x", "y"))

poi.sport.entertainment <- Get.Sport.Entertainment.Pois(LDN.GRID.SPDF) %>%
  dplyr::mutate(log.poi.sport.entertainment.count = log(1 + poi.sport.entertainment.count))
merged.data <- dplyr::full_join(merged.data, poi.sport.entertainment, by = c("x", "y"))

poi.accommodation <- Get.Accommodation.Pois(LDN.GRID.SPDF) %>%
  dplyr::mutate(log.poi.accommodation.count = log(1 + poi.accommodation.count))
merged.data <- dplyr::full_join(merged.data, poi.accommodation, by = c("x", "y"))

poi.edu.health <- Get.Education.Health.Pois(LDN.GRID.SPDF) %>%
  dplyr::mutate(log.poi.edu.health.count = log(1 + poi.edu.health.count))
merged.data <- dplyr::full_join(merged.data, poi.edu.health, by = c("x", "y"))

# aggregate POIs
merged.data <- merged.data %>%
  dplyr::mutate(poi.all = poi.retail.count + poi.eatdrink.count + poi.sport.entertainment.count + 
                          poi.accommodation.count + poi.edu.health.count) %>%
  dplyr::mutate(log.poi.all = log(1 + poi.all))

ethnic.heterogeneity <- Get.Ethnic.Heterogeneity(LDN.GRID.SPDF, end.date, soc.econ.folder, geom.name = "lsoa")
merged.data <- dplyr::full_join(merged.data, ethnic.heterogeneity, by = c("x", "y"))

transport.accessibility <- Get.Transport.Accessibility(LDN.GRID.SPDF, end.date, soc.econ.folder) %>%
  dplyr::mutate(log.accessibility = log(accessibility))
merged.data <- dplyr::full_join(merged.data, transport.accessibility, by = c("x", "y"))

population.turnover <- Get.Population.Turnover(LDN.GRID.SPDF, end.date, soc.econ.folder, "oa") %>%
  dplyr::mutate(people.moved.in.out = people.moved.in + people.moved.out) %>%
  dplyr::mutate(population.turnover = (people.moved.in + people.moved.out) / people.current) %>%
  dplyr::mutate(log.people.moved.in = log(1 + people.moved.in)) %>%
  dplyr::mutate(log.people.moved.out = log(1 + people.moved.out)) %>%
  dplyr::mutate(log.people.moved.in.out = log(1 + people.moved.in.out)) %>%
  dplyr::mutate(log.population.turnover = log(1 + population.turnover))
merged.data <- dplyr::full_join(merged.data, population.turnover, by = c("x", "y"))

dwelling.types <- Get.Dwelling.Type.Data(LDN.GRID.SPDF, end.date, soc.econ.folder, geom.name = "oa") %>%
  dplyr::mutate(detached.semidetached.houses.fraction = detached.houses.fraction + semidetached.houses.fraction) %>%
  dplyr::mutate(log.all.houses.fraction = log(1 + all.houses.fraction)) %>%
  dplyr::mutate(log.detached.houses.fraction = log(1 + detached.houses.fraction)) %>%
  dplyr::mutate(log.semidetached.houses.fraction = log(1 + semidetached.houses.fraction)) %>%
  dplyr::mutate(log.terraced.houses.fraction = log(1 + terraced.houses.fraction)) %>%
  dplyr::mutate(log.detached.semidetached.houses.fraction = log(1 + detached.semidetached.houses.fraction)) %>%
  dplyr::mutate(detached.semidetached.houses.count = detached.houses.count + semidetached.houses.count) %>%
  dplyr::mutate(log.all.houses.count = log(1 + all.houses.count)) %>%
  dplyr::mutate(log.detached.houses.count = log(1 + detached.houses.count)) %>%
  dplyr::mutate(log.semidetached.houses.count = log(1 + semidetached.houses.count)) %>%
  dplyr::mutate(log.terraced.houses.count = log(1 + terraced.houses.count)) %>%
  dplyr::mutate(log.detached.semidetached.houses.count = log(1 + detached.semidetached.houses.count))
merged.data <- dplyr::full_join(merged.data, dwelling.types, by = c("x", "y"))

occupation.variation <- Get.Skill.Occupation(LDN.GRID.SPDF, end.date, soc.econ.folder, "lsoa")
merged.data <- dplyr::full_join(merged.data, occupation.variation, by = c("x", "y"))

housing.tenure.data <- Get.House.Tenure(LDN.GRID.SPDF, end.date, soc.econ.folder, "oa") %>%
  dplyr::mutate(log.tenure.owned.fraction = log(1 + tenure.owned.fraction)) %>%
  dplyr::mutate(log.tenure.rented.social.fraction = log(1 + tenure.rented.social.fraction)) %>%
  dplyr::mutate(log.tenure.rented.private.fraction = log(1 + tenure.rented.private.fraction)) %>%
  dplyr::mutate(log.tenure.other.fraction = log(1 + tenure.other.fraction)) %>%
  dplyr::mutate(log.tenure.owned.count = log(1 + tenure.owned.count)) %>%
  dplyr::mutate(log.tenure.rented.social.count = log(1 + tenure.rented.social.count)) %>%
  dplyr::mutate(log.tenure.rented.private.count = log(1 + tenure.rented.private.count)) %>%
  dplyr::mutate(log.tenure.other.count = log(1 + tenure.other.count))
merged.data <- dplyr::full_join(merged.data, housing.tenure.data, by = c("x", "y"))

household.type.data <- Get.Household.Types(LDN.GRID.SPDF, end.date, soc.econ.folder, "oa") %>%
  dplyr::mutate(log.single.parent.household.fraction = log(1 + single.parent.household.fraction)) %>%
  dplyr::mutate(log.one.person.household.fraction = log(1 + one.person.household.fraction)) %>%
  dplyr::mutate(log.couple.with.children.fraction = log(1 + couple.with.children.fraction)) %>%
  dplyr::mutate(log.single.parent.household.count = log(1 + single.parent.household.count)) %>%
  dplyr::mutate(log.one.person.household.count = log(1 + one.person.household.count)) %>%
  dplyr::mutate(log.couple.with.children.count = log(1 + couple.with.children.count))
merged.data <- dplyr::full_join(merged.data, household.type.data, by = c("x", "y")) # 

household.income <- Get.Mean.HH.Income(LDN.GRID.SPDF, end.date, soc.econ.folder) %>%
  dplyr::mutate(log.mean.hh.income = log(mean.hh.income))
merged.data <- dplyr::full_join(merged.data, household.income, by = c("x", "y"))

house.prices <- Get.House.Prices(LDN.GRID.SPDF, end.date, soc.econ.folder) %>%
  dplyr::mutate(log.house.price = log(house.price))
merged.data <- dplyr::full_join(merged.data, house.prices, by = c("x", "y"))

median.age.data <- Get.Age.Data(LDN.GRID.SPDF, end.date, soc.econ.folder, 'oa') %>%
  dplyr::mutate(log.median.age = log(median.age))
merged.data <- dplyr::full_join(merged.data, median.age.data, by = c("x", "y"))

urbanisation.data <- Get.Urbanisation.Level(LDN.GRID.SPDF)
merged.data <- dplyr::full_join(merged.data, urbanisation.data, by = c("x", "y"))

merged.data['intercept'] <- 1

###############################################################################
#                                 Publishing                                  #
###############################################################################
print("Adding LSOA/MSOA information")
temp <- sp::over(LDN.GRID.SPDF, LDN.LSOA.SPDF)
region.labels <- data.frame(coordinates(gCentroid(LDN.GRID.SPDF, byid=TRUE)), temp[, c('LSOA11CD', 'MSOA11CD', 'LAD11CD')])
merged.data <- base::merge(x=merged.data, y=region.labels, by=c("x", "y"), all.x = T)

print("Adding Ward information")
temp <- sp::over(LDN.GRID.SPDF, LDN.WARD.SPDF)
ward.labels <- data.frame(coordinates(gCentroid(LDN.GRID.SPDF, byid=TRUE)), temp[, c('GSS_CODE')])
names(ward.labels) <- c("x", "y", "WARD")
merged.data <- base::merge(x=merged.data, y=ward.labels, by=c("x", "y"), all.x = T)


print("Publishing the dataset")
output.file.name <- paste("grid-raw-", collection.unit, "-", cov.method , "-spatial-counts-covariates-", month(start.date), year(start.date), "-", 
                          month(end.date), year(end.date), "-", resolution, ".csv", sep="")
write.csv(x = merged.data, file = file.path(datafolder, "processed", output.file.name))


print("Publishing the mapping from LSOA/MSOA labels to the respective centroids")
# Generate Centroids for the geometries
centroids.lsoa <- coordinates(gCentroid(LDN.LSOA.SPDF, byid=TRUE))
centroids.msoa <- coordinates(gCentroid(LDN.MSOA.SPDF, byid=TRUE))
centroids.ward <- coordinates(gCentroid(LDN.WARD.SPDF, byid=TRUE))
centroids.lad <- coordinates(gCentroid(LDN.LAD.SPDF, byid=TRUE))

labels.lsoa <- (LDN.LSOA.SPDF@data)['LSOA11CD']
labels.msoa <- (LDN.MSOA.SPDF@data)['MSOA11CD']
labels.lad <- (LDN.LAD.SPDF@data)['LAD11CD']
labels.ward <- (LDN.WARD.SPDF@data)['GSS_CODE']

LDN.WARD.SPDF@data
spplot(LDN.WARD.SPDF, zcol='HECTARES')

write.csv(x=data.frame(labels.lsoa, centroids.lsoa), file=file.path(datafolder, "processed", "lsoa-centroids-map.csv"))
write.csv(x=data.frame(labels.msoa, centroids.msoa), file=file.path(datafolder, "processed", "msoa-centroids-map.csv"))
write.csv(x=data.frame(labels.ward, centroids.ward), file=file.path(datafolder, "processed", "ward-centroids-map.csv"))


print("Publishing the data as a shape file.")
merged.data.spdf <- merged.data
coordinates(merged.data.spdf) <- ~x+y
proj4string(merged.data.spdf) <- proj4string(LDN.GRID.SPDF)

merged.data.grid.df <- sp::over(x = LDN.GRID.SPDF, merged.data.spdf)
merged.data.grid.spdf <- LDN.GRID.SPDF
merged.data.grid.spdf@data <- merged.data.grid.df


output.file.name <- paste("grid-raw-", collection.unit, "-", cov.method , "-spatial-counts-covariates-", month(start.date), year(start.date), "-", 
                          month(end.date), year(end.date), "-", resolution, ".gpk", sep="")
writeOGR(merged.data.grid.spdf, 
         dsn = file.path(datafolder, 'processed', output.file.name),
         layer = 'processed_data',
         driver = 'GPKG')




counts.histogram <- hist(as.matrix(all.crime['burglary']), breaks=50, plot=F)
hist.breaks <- counts.histogram$breaks
hist.bucket.centroids <- head((hist.breaks + hist.breaks[-1]) / 2, -1)
hist.data.for.plots <- cbind(hist.bucket.centroids, counts.histogram$counts)
write.csv(hist.data.for.plots, file.path("../../reports", "paper_plots",
                                         paste("counts_histogram_data_", 
                                               month(start.date), 
                                               year(start.date), 
                                               "-", 
                                               month(end.date), 
                                               year(end.date), ".dat",
                                               sep = "")),
          row.names=F, col.names=c('x', 'count'))