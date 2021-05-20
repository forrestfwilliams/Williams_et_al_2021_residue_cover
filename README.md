# Estimating crop residue cover utilizing alternate ground truth measurements and multi-satellite regression models

Soil erosion within agricultural landscapes has a significant impact on both environmental and economic systems and is strongly driven by low levels of residue cover. Soil erosion models such as the Water and Erosion Prediction Project (WEPP) model and its large area implementation, the Daily Erosion Project, are important tools for understanding patterns of soil erosion. To produce these predictions however, soil erosion models depend on the accurate estimation of crop residue cover over large regions to infer tillage practices. Remote sensing analyses are now becoming accepted to estimate crop residue cover with field level resolution, but training these models typically requires specially developed training datasets that are not practical outside of small study areas. An alternative source of training data may be commonly conducted tillage surveys that capture information via rapid 'windshield' surveys. We evaluate multiple survey types and lo-cations, twelve spectral indices sourced from four satellite platforms, and three distinct regression functions, were used to create the best performing model. Multi-sensor and multi-image analysis were conducted within the Google Earth Engine platform to produce regression models that use remote sensing data to predict the coverage of crop residue. Crop residue 'windshield' surveys based on identifying four tillage classes, and 10% crop residue bin surveys were found to produce highly variable training data that were not suitable for regression analyses, but in-field photog-raphy analyzed using point count surveys did produce reliable training data. Overall, a bivariate asymptotic growth regression model that utilized the Landsat 7 NDTI index and the Sentinel-1 VV polarization band was found to be the most reliable predictor of crop residue cover, having a RMSE of 16.6%. Landsat 7 NDVI was the best proxy for emergent vegetation, and Sentinel-1 VV amplitude was the best proxy for soil moisture or soil roughness. This study demonstrates that where quality training data exists, the Google Earth Engine platform can be used to reliably estimate crop residue cover over large regions in the Midwestern United States.