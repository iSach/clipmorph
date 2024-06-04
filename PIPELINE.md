# Pipeline

## Training Docker Container

> *Docker file:* `Dockerfile.train`

We package our model training script into a Docker container. Similar to 
deployment and serving, we copy necessary files and install dependencies.

To build the Docker image, run:
`docker build -t clipmorph_train -f Dockerfile.train .`

To train a model, it is required to:
1. Prepare a data folder that contains the reference image named `style.jpg` and a `visual_genome` folder that contains the Visual Genome 
   dataset. One can download this data using `training_data/download_genome.
   sh`.
3. Run:

`sudo docker run --gpus all -v ./models:/workspace/models -v ./data:/workspace/data --rm -it clipmorph_train`

This will start the training job that will get logged on W&B, and the model 
will be saved in the mounted `models` folder.

If you do not wish to use the interactive mode, Weights and Biases will not work as such. You need to give your API key when running Docker:

`sudo docker run -e WANDB_API_KEY=<your-key> --gpus all -v ./models:/workspace/models -v ./data:/workspace/data --rm -it clipmorph_train`

> [!TIP]
> Find more details and screenshots about our use of Weights & Biases in [DEVELOPMENT.md](DEVELOPMENT.md).

## Cloud Training & Pipeline

We were **not** able to deploy training to the cloud due to a lack of GPUs 
and issues with Vertex training. Consequently, we were not able to create a 
pipeline on the cloud either. However, we tested the Docker image locally, the container runs without issues. It could thus theoretically be used with any cloud provider.
