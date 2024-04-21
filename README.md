<div align="center">
  <img src=".github/assets/clipmorph_logo.png" alt="ClipMorph Logo" 
width="200"/>
    <br>
    <em>Make your clips come to life!</em>
</div><br>

<div align="center">
    <a href='https://github.com/iSach/clipmorph/actions/workflows/clipmorph_tests.yml'>
        <img src='https://github.com/iSach/clipmorph/actions/workflows/clipmorph_tests.yml/badge.svg' alt='Test 
Status' /></a>
    <a href='https://github.com/iSach/clipmorph/actions/workflows/deploy.yml'>
        <img src='https://github.com/iSach/clipmorph/actions/workflows/deploy.yml/badge.svg' 
alt='Deploy Status' /></a>
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
- `api`: contains the Flask Web API for remotely calling the model.
- `models/`: contains pre-trained models (weights), can be loaded with `run.py`.
- `tests/`: contains unit tests for the project.
- `training_data/`: contains style images and a script to download Visal 
  Genome.

## Model Development

Please see [DEVELOPMENT.md](DEVELOPMENT.md).

<u>TL;DR</u>: Our model performs neural style transfer. It is a deep 
U-Net-style 
network (image-to-image with downsampling and upsampling). It learns to 
apply a style by minimizing special losses that force the content to stay 
the same but the style to match another image. We collect a dataset of diverse 
images (VisualGenome) and several well-known artworks for training. One 
model is trained per style. We qualitatively validated our model and tracked 
training status using Weights & Biases.

## Model Deployment

Please see [DEPLOYMENT.md](DEPLOYMENT.md).

<u>TL;DR</u>: 

## Model Pipeline

Please see [PIPELINE.md](PIPELINE.md).

<u>TL;DR</u>:

## CI / CD

Please see [CICD.md](CICD.md).

<u>TL;DR</u>: We perform continuous integration (CI) through GitHub Actions 
with unit tests (using PyTest), as well as code style tests (using Ruff). We 
also perform continuous deployment (CD) of our Flask Web API to Google Cloud 
Run.
