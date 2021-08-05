#!/bin/bash
chmod +x ./release.sh

python3 unzip.py jump.zip
rm jump.zip
cd jump
../release.sh
cd ..

python3 unzip.py walk.zip
rm walk.zip
cd walk
../release.sh
cd ..

python3 unzip.py wave.zip
rm wave.zip
cd wave
../release.sh
cd ..

python3 unzip.py run.zip
rm run.zip
cd run
../release.sh
cd ..