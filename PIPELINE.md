# Pipeline

TODO.

- Package training script in Docker to run locally (extremely easy)
  - We just need to make the Dockerfile with cuda image
  - install dependencies etc
  - and find how it should be started (like do we have to enter the 
    container and run the python script or should it run automatically idk)
  - -> Check Nerfstudio docker image
- Try training at least one model online. Will be hard to train automatically
  - Vertex training or if we're lucky compute instance.
- Pipeline: probably will be too hard (Vertex Pipeline or Docker compose)



As we're probably not gonna train on the cloud, we should at least detail 
our methodology for training the models, i.e., locally :)