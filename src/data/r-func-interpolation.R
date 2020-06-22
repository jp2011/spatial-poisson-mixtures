libraries <- c("rgdal",
               "rgeos",
               "spatstat",
               "akima",
               "raster",
               "dplyr",
               "sf"
)
lapply(libraries, require, character.only = TRUE)

Simple.Overlay <- function(desired.spdf, measurement.spdf, covariate.names) {
  desired.data <- sp::over(desired.spdf, measurement.spdf)
  output <- desired.spdf
  output@data <- desired.data[covariate.names]
  return(output)
}

Weighted.Overlay <- function(desired.spdf, measurement.spdf, covariate.names,
                             extensive = FALSE){
  source.sf <- st_as_sf(measurement.spdf)
  target.sf <- st_as_sf(desired.spdf)
 
  source.sf <- source.sf %>%
    dplyr::select(covariate.names)

  output.sf <- st_interpolate_aw(source.sf, target.sf, extensive = extensive)
  output.spdf <- as(output.sf, "Spatial")
  return(output.spdf) 
}

Triangular.Interpolation <- function(desired.spdf, measurements.spdf, covariate.names) {
  # the desired spdf needs to be a grid
  
  grid.centroids.df <- data.frame(gCentroid(desired.spdf, byid=TRUE))
  
  x.for.interpolation <- sort(unique(grid.centroids.df$x))
  y.for.interpolation <- sort(unique(grid.centroids.df$y))
  
  measurement.locations.df <- data.frame(gCentroid(measurements.spdf, byid=TRUE))
  
  all.desired.interp.df <- data.frame(grid.centroids.df)
  for (cov.name in covariate.names) {
    
    interpolated.grid <- akima::interp(x = measurement.locations.df$x,
                                       y = measurement.locations.df$y,
                                       z = (measurements.spdf@data)[[cov.name]],
                                       xo = x.for.interpolation,
                                       yo = y.for.interpolation,
                                       extrap=FALSE,
                                       linear = TRUE,
                                       duplicate="strip")
    
    interpolated.df <- data.frame(expand.grid("x"=interpolated.grid$x, "y"=interpolated.grid$y))
    interpolated.df[cov.name] <- c(interpolated.grid$z)
    
    # choose only the rows that are in the desired geometry and append it to the df with desired columns
    all.desired.interp.df <- base::merge(all.desired.interp.df, interpolated.df, by=c("x", "y"), all.x=TRUE, sort=FALSE)
  }
  
  output <- desired.spdf
  output@data <- all.desired.interp.df
  return(output)
}

Gaussian.Smooth <- function(desired.spdf, measurements.spdf, covariate.names, cell.size=500) {
  smoothing.resolution <- cell.size / 1
  smoothing.sd <- cell.size / 2
  measurements.extent <- extent(measurements.spdf)
  x <- seq(measurements.extent[1],measurements.extent[2],by=smoothing.resolution) 
  y <- seq(measurements.extent[3],measurements.extent[4],by=smoothing.resolution)
  
  smoothing.grid <- expand.grid(x=x,y=y)
  coordinates(smoothing.grid) <- ~x+y
  gridded(smoothing.grid) <- TRUE
  proj4string(smoothing.grid) <- proj4string(measurements.spdf)
  
  smoothing.grid.spdf <- as(as(smoothing.grid, 'SpatialPolygons'), 'SpatialPolygonsDataFrame')   # convert into polygon
  smoothing.grid.centroids <- gCentroid(smoothing.grid.spdf, byid=TRUE)
  
  smoothing.data.overlay <- sp::over(smoothing.grid.centroids, measurements.spdf)
  smoothing.data.overlay <- smoothing.data.overlay[covariate.names]
  smoothing.data.overlay[is.na(smoothing.data.overlay)] <- 0
  
  smoothed.data.full <- data.frame(smoothing.grid.centroids@coords)
  for(cov.name in covariate.names) {
    im.xyz <- data.frame(smoothing.grid.centroids@coords, z=smoothing.data.overlay[cov.name])
    smooth.data.im <- spatstat::blur(as.im(im.xyz), smoothing.sd, bleed=TRUE)
    smooth.data.matrix <- as.matrix.im(smooth.data.im)
    smooth.data.vector <- as.vector(t(smooth.data.matrix))
    smoothed.data.full[cov.name] <- smooth.data.vector
  }
  smoothing.grid.spdf@data <- smoothed.data.full
  
  # project to the desired geometry
  output.spdf <- desired.spdf
  output.spdf@data <- sp::over(desired.spdf, smoothing.grid.spdf)
  return(output.spdf)
}
