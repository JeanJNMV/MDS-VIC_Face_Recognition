# -------- ORL (ATT Faces) --------
wget https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip

mkdir -p data/ORL
unzip att_faces.zip -d data/ORL

# Move contents up one level
mv data/ORL/att_faces/* data/ORL/
rm -r data/ORL/att_faces
rm att_faces.zip


# -------- Yale Face Database --------
uvx kaggle datasets download olgabelitskaya/yale-face-database

mkdir -p data/Yale
unzip yale-face-database.zip -d data/Yale

# Move contents up one level
mv data/Yale/data/* data/Yale/
rm -r data/Yale/data
rm yale-face-database.zip
