# wget https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
# mkdir -p data/ORL
# unzip att_faces.zip -d data/ORL
# rm att_faces.zip

uvx kaggle datasets download olgabelitskaya/yale-face-database
mkdir -p data/Yale
unzip yale-face-database.zip -d data/Yale
mv data/Yale/data/* data/Yale/
rm -r data/Yale/data
rm yale-face-database.zip
