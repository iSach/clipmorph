# Download the Visual Genome dataset
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip

# Unzip the dataset and remove the zip file
unzip images.zip
rm images.zip

# Move "VG_100K" into "training_data/visual_genome"
mv VG_100K training_data/visual_genome

echo "Done."