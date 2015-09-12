Extracts surface heat flux values from SMU geothermal heat flux map found at <http://www.smu.edu/~/media/Site/Dedman/Academics/Programs/Geothermal%20Lab/Graphics/SMUHeatFlowMap2011_CopyrightVA0001377160_jpg.ashx?la=en>

Run script AllHeatProd.py to extract surface heat flux data and produce CSV file of extracted values and map image.  Script approximates 49th parallel as quadratic function, several US state borders as meridian references. It hen uses a 2D tree nearest neighbor to solve for longitude and a Newton solver to compute distance to the 49th parallel parabola. Color values are then matched to the original colorbar to assign values of heat flux at each location. 

Outputs will be written to 'LongLatSurfaceHeat.csv' and 'HeatMapUSAr.png'.