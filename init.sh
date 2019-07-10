pip install -r requirements.txt

mkdir data
cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz
tar -xvzf imagenette-160.tgz
cd ..

mkdir weights
mkdir runs
