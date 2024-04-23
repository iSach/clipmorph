# Pipeline

## Training Docker Container

> *Docker file:* `Dockerfile.train`

We package our model training script into a Docker container. Similar to 
deployment and serving, we copy necessary files and install dependencies.

To build the Docker image, run:
`docker build -t clipmorph_train -f Dockerfile.train .`

To train a model, it is required to:
1. Prepare a data folder that contains the reference image named `style.
   png` and a `visual_genome` folder that contains the Visual Genome 
   dataset. One can download this data using `training_data/download_genome.
   sh`.
2. Set up Weights & Biases for tracking.
3. Run:

`wandb docker run -v /path/data:/workspace/data -v 
/path/output_models:/workspace/models clipmorph_train`

This will start the training job that will get logged on W&B, and the model 
will be saved in the mounted `output_models` folder.

## Cloud Training & Pipeline

We were **not** able to deploy training to the cloud due to a lack of GPUs 
and issues with Vertex training. Consequently, we were not able to create a 
pipeline on the cloud either.