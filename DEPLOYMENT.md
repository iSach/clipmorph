# Deployment

TODO.

If we manage to run on GPU:
- Make Flask API CPU only that calls the GPU model
- eg Vertex Predict

If we don't manage to run on GPU:
- Merge Flask API with the model (remove api folder & make everything in 
  the main folder I think, we one Dockerfile per thing maybe idk)
- Make a Dockerfile that runs the Flask API
- and run on CPU on GCR
- -> Deploy to GCR automatically with GitHub Actions maybe.


Anyway, we'll need our docker flask + deploy.
