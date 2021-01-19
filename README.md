

Model training perfromance    |  1-hour ahead predicions 
:-------------------------:|:-------------------------:
![training_vs_test_data](https://user-images.githubusercontent.com/16349565/105021136-7dd22000-5a48-11eb-83a3-57ffb2b9101b.png)  | ![make_predictions](https://user-images.githubusercontent.com/16349565/105020775-08fee600-5a48-11eb-8e70-540e4a7c3ad3.png)


# FORESEE_ETHZ

Perform a classification/regression using a Random Forest (RF):
- We predict a target traffic intensity level in a unit location at a specific time interval. 
- We adopt the following types of features: 
  1) time features, such as hour, day-of-week, and week; 
  2) spatial features, such as location_id; 
  3) rain (precipitation); 
  4) traffic features such as average hourly traffic speed
  5) Context such as holidays, sporting events, construction, etc.  

Other input features that we could consider including are:
- geographical distances between locations or with respect to special areas (attractions, tourist spots, etc.) - Although Forest-based Classification and Regression is not a spatial machine learning tool, one way to leverage the power of space in the  analysis is using distance features
- quality of road (bad, good, etc.)


# INSTRUCTIONS
- To train RF models run the following file: train_traffic_intensity_RF_model.py
- To make predictions using saved model (model saved to disk using Pickle), run the following file:  make_predictions.py
- Below are instructions to build Docker image


# BUILD AND RUN DOCKER IMAGE

- Build a docker image (include the dot at the end): docker build -t foresee-rf-app .

- Run a docker image: docker run --rm -it foresee-rf-app

- To check out the actual content of the docker image (i.e. files copied into the image), run the following: docker run -it foresee-rf-app bash


My docker resources:
- https://hub.docker.com/search?q=&type=image
- https://gist.github.com/adamveld12/4815792fadf119ef41bd
- https://docs.docker.com/engine/reference/builder/


# HOW TO PUSH A DOCKER IMAGE TO DOCKERHUB
- docker images (to see the dockerimage-id)
- docker tag <dockerimage-id> <dockerhub-user-id>/foresee-rf-app
- docker login (successful)
- docker push <dockerhub-id>/foresee-rf-app



