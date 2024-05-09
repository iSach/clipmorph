<div align="center">
  <img src=".github/assets/clipmorph_logo.png" alt="ClipMorph Logo" 
width="200"/>
    <br>
    <em>Make your clips come to life!</em>
</div><br>

<div align="center">
    <a href="https://clipmorph.isach.be">
        <img alt="License" src="https://img.shields.io/badge/Web App-Online-aqua.svg"></a>
    <a href='https://github.com/iSach/clipmorph/actions/workflows/clipmorph_tests.yml'>
        <img src='https://github.com/iSach/clipmorph/actions/workflows/clipmorph_tests.yml/badge.svg' alt='Test 
Status' /></a>
    <a href='https://github.com/iSach/clipmorph/actions/workflows/code_style.yml'>
        <img src='https://github.com/iSach/clipmorph/actions/workflows/code_style.yml/badge.svg' 
alt='Code Style Status' /></a>
    <a href="https://github.com/iSach/clipmorph/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</div>

## About

ClipMorph is an AI-powered stylization tool to morph your clips into a 
different style. It offers a wide range of pre-trained styles to choose from,
or new styles can be effortlessly added by training a new model in under 
one hour. Built on state-of-the-art image-to-image AI models, ClipMorph can 
stylize videos in real-time (~25-30 frames per second).

The team behind ClipMorph is composed of:
- [Sacha Lewin](https://github.com/iSach)
- [Axelle Schyns](https://github.com/AxelleSchyns)
- [Laurie Boveroux](https://github.com/LaurieBvrx)
- [Arthur Louette](https://github.com/LouetteArthur)

Please find a more detailed description of the product itself [here](canvas.pdf).

> _This project is developed as part of the course "Machine Learning Systems 
Design" @ ULiège ([INFO9023](https://github.com/ThomasVrancken/info9023-mlops))._

## Repository Structure

This repository contains the code for training, inference, and deployment, 
as well as pre-trained styles, training data scripts, and more.

- `clipmorph/`: contains the main codebase for the project.
  - `run.py`: script for running the application on an image/video.
  - `train.py`: script to train a new model.
- `app`: contains the Flask Web API for remotely calling the model.
- `models/`: contains pre-trained models (weights), can be loaded with `run.py`.
- `tests/`: contains unit tests for the project.
- `training_data/`: contains style images and a script to download Visual 
  Genome.

## Model Development

Please see [DEVELOPMENT.md](DEVELOPMENT.md).

<u>TL;DR</u>: Our model performs neural style transfer. It is a deep 
U-Net-style 
network (image-to-image with downsampling and upsampling). It learns to 
apply a style by minimizing specific loss functions that force the content to remain 
the same while the style has to match the one of another reference image. We collect a dataset of diverse 
images (VisualGenome) and several well-known artworks for training. One 
model is trained per style. We qualitatively validated our model and tracked 
training status using Weights & Biases.

## Model Deployment

Please see [DEPLOYMENT.md](DEPLOYMENT.md).

<u>TL;DR</u>: A Flask web interface was built to serve the model, allowing users to upload files and receive stylized results.
A Docker container was created to package the code, models, and 
dependencies. The container is automatically deployed to Google Cloud Run, a 
serverless platform, for continuous deployment (CD). Users can upload videos or 
images and receive predictions from pre-trained models through the app at https://clipmorph.isach.be. 
Due to GPU unavailability on Google Cloud Run, the model was 
deployed as a CPU-only instance, strongly limiting performance for long videos. 

## Model Pipeline

Please see [PIPELINE.md](PIPELINE.md).

<u>TL;DR</u>: We create a Docker container to package the model training 
script. To train a model, we provide instructions on how to run the Docker 
image once it's built. It logs training to Weights&Biases and outputs the 
pre-trained model. Due to a lack of GPUs and issues with Vertex training, 
we were unable to deploy training to the cloud or create a cloud-based pipeline, 
but this Docker image was tested locally and could theoretically be used with any cloud provider.

## CI / CD

Please see [CICD.md](CICD.md).

<u>TL;DR</u>: We perform continuous integration (CI) through GitHub Actions 
with unit tests (using PyTest), as well as code style tests (using Ruff). These tests
are run when pushing to the dev or the main branch, as well as when submitting a PR into these
branches to avoid merging commits that do not respect the guidelines or break the codebase. We 
also perform continuous deployment (CD) of our Flask Web API to Google Cloud Run.

## Future Work

This project was a nice way to discover ML systems design. It notably 
taught us the various challenges of deploying a model to production, and 
there are several things we were not unfortunately able to implement, due to time and 
resource constraints (GPUs). Here are some ideas for further improving the 
project:
- Implement a pipeline that allows users to upload their styles so that then, 
  a model is trained automatically.
  - Store pre-trained models in a Cloud Storage bucket and load them 
    dynamically from the app, instead of storing them in the repo/acontainer.
  - Make the pipeline use GPU for training.
- Make the app run predictions on GPU for real-time processing.
- Cache models in memory for faster processing. An example method is to 
  keep as many models as possible in memory and only unload them when 
  it is required to load new ones and memory is full. This would significantly reduce the latency of new requests to 
  the app.
- Improve the UI/UX and features of the app: e.g., fully working progress bar with another worker for the running task, displaying 
  the video on the website instead of downloading, history of processed 
  videos, ETA, etc.
- Monitoring/Dashboarding of our application.
