# FORESEE_ETHZ

Perform a classification/regression using a Random Forest:
- We predict a target traffic intensity level in a unit location at a specific time interval. 
- We adopt the following types of features: 
  1) time features, such as hour, day-of-week, and week; 
  2) spatial features, such as location_id; 
  3) rain (precipitation); 
  4) traffic features such as average hourly traffic speed
  5) Context such as holidays, sporting events, construction, etc.  

Other input features that we could consider including are:
- geographical distances between locations or with respect to special areas (attractions, tourist spots, etc.) - Although Forest-based Classification and Regression is not a spatial machine learning tool, one way to leverage the power of space in the  analysis is using distance features
- Quality of road (bad, good, etc.)
