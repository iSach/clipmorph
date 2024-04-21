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

> _This project is developed as part of the course "Machine Learning Systems 
Design" at the University of Liège ([INFO9023](https://github.
com/ThomasVrancken/info9023-mlops))._

## Model Development

## Model Deployment

## Model Pipeline

## Canvas

The value proposition canvas can be found as `canvas.pdf` in the root folder. 

## Code

The code is contained in `climorph.ipynb`. It contains instructions to run each part (training or applying).

Training with about 65k images (VisualGenome) resized to 512x512 takes about 2 hours on an RTX 3080. 
If you only want to apply one of the models stored in models, run all cells except the training one! 
Stylization can be done in the last cell, for an image or a GIF/mp4.

Frames are stored temporarily in the frames_video and frames_video_stylized folders. These should not really be considered.

For both parts of the project, content is stored in `content`, style images in `style`, and the results are stored in `result`.

