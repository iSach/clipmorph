# ClipMorph Web Application

This folder contains the code for the Web application for ClipMorph, which allows to apply pre-trained styles to any image or video remotely.

Our app was written with Flask in Python, and consists of a simple web page that allows to select the style and upload media to stylize.

<div align="center">
  <img src="../.github/assets/website_snapshot.png" alt="ClipMorph Web App" 
width="900"/>
    <br>
    <p>Example screenshot of the web application.</p>
</div><br>

### Docker Container

As detailed in [CICD.md](../CICD.md), our app is continuously deployed to Google Cloud Run. No Github Action file was manually written as GCR handles this for us through their interface.
