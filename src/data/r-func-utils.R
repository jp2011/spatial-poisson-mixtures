libraries <- c(
  "lubridate",
  "raster",
  "rgdal",
  "rgeos"
)
invisible(lapply(libraries, library, character.only = TRUE))


ym <- function(ymd_object) {
  paste(year(ymd_object),format.Date(ymd_object, "%m"), sep="-")
}


Get.Crime.Name <- function(crime.type.token) {
  crime.type.names <- c("Violence and sexual offences", "Anti-social behaviour", "Burglary", "Criminal damage and arson", "Other theft", "Possession of weapons", "Robbery", "Theft from the person", "Vehicle crime", "Other crime", "Public order", "Shoplifting", "Drugs", "Bicycle theft", "Violent crime", "Public disorder and weapons")
  crime.type.tokens <- c("violence-and-sexual-offences", "anti-social-behaviour", "burglary", "criminal-damage-and-arson", "other-theft", "possession-of-weapons", "robbery", "theft-from-the-person", "vehicle-crime", "other-crime", "public-order", "shoplifting", "drugs", "bicycle-theft", "violent-crime", "public-disorder-and-weapons")
  names(crime.type.names) <- crime.type.tokens
  return(crime.type.names[crime.type.token])
}


Get.All.Crime.Types.Tokens <- function() {
  crime.type.tokens <- c("violence-and-sexual-offences", "anti-social-behaviour", "burglary", "criminal-damage-and-arson", "other-theft", "possession-of-weapons", "robbery", "theft-from-the-person", "vehicle-crime", "other-crime", "public-order", "shoplifting", "drugs", "bicycle-theft", "violent-crime", "public-disorder-and-weapons")
  return(crime.type.tokens)
}


Get.Closest.Available.Year.Index <- function(current.year, available.years) {
  current.year.num <- as.numeric(current.year)
  available.years.num <- as.numeric(available.years)
  distances <- sapply(available.years.num, function(available.year) {abs(current.year.num - available.year)})
  min.index <- which.min(distances)
  return(min.index)
}


####################################  GEO THINGS #################################### 
BNG.CRS.PROJECTION.CODE <- "+init=epsg:27700"
GPS.CRS.PROJECTION.CODE <- "+init=epsg:4326"

LDN.OA.SPDF <- readOGR(dsn=file.path("../../data", "external", "ESRI", "OA_2011_London_gen_MHW.shp"))
LDN.MSOA.SPDF <- readOGR(dsn=file.path("../../data", "external", "ESRI", "MSOA_2011_London_gen_MHW.shp"))
LDN.LSOA.SPDF <- readOGR(dsn=file.path("../../data", "external", "ESRI", "LSOA_2011_London_gen_MHW.shp"))
LDN.WARD.SPDF <- readOGR(dsn=file.path("../../data", "external", "ESRI", "London_Ward_CityMerged.shp"))
LDN.LAD.SPDF <- aggregate(LDN.LSOA.SPDF, by=c("LAD11CD"), dissolve=F)


NOMIS.OA.CODE <- '2013265927TYPE299'
NOMIS.LSOA.CODE <- '2013265927TYPE298'
NOMIS.MSOA.CODE <- '2013265927TYPE297'

Get.Ldn.Grid.Spdf <- function(resolution, geom.to.use=LDN.MSOA.SPDF) {
  Extent<- extent(geom.to.use) #this is the geographic extent of the grid. It is based on the London object.
  x <- seq(Extent[1],Extent[2],by=resolution)  # where resolution is the pixel size you desire
  y <- seq(Extent[3],Extent[4],by=resolution)
  xy <- expand.grid(x=x,y=y)
  coordinates(xy) <- ~x+y
  gridded(xy) <- TRUE
  proj4string(xy) <- proj4string(geom.to.use)
  g.ldn.grid <- xy[geom.to.use]
  g.ldn.grid <- as(as(g.ldn.grid, 'SpatialPolygons'), 'SpatialPolygonsDataFrame')   # convert into polygon
  return(g.ldn.grid)
}

Get.Geometry.By.Name <- function(geom.name) {
  geom.names <- c("oa", "msoa", "lsoa", "ward")
  geom.objects <- c(LDN.OA.SPDF, LDN.MSOA.SPDF, LDN.LSOA.SPDF, LDN.WARD.SPDF)
  names(geom.objects) <- geom.names
  return(geom.objects[[geom.name]])
}

Get.Geo.Id.Column.Name <- function(geom.name) {
  geom.names <- c("oa", "msoa", "lsoa")
  geom.objects <- c("OA11CD", "MSOA11CD", "LSOA11CD")
  names(geom.objects) <- geom.names
  return(geom.objects[[geom.name]])
}

Get.Geo.Nomis.Code <- function(geom.name) {
  geom.names <- c("oa", "msoa", "lsoa")
  geom.objects <- c(NOMIS.OA.CODE, NOMIS.MSOA.CODE, NOMIS.LSOA.CODE)
  names(geom.objects) <- geom.names
  return(geom.objects[[geom.name]])
}


