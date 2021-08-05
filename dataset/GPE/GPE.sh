#!/bin/bash
python3 unzip.py 1-person.zip
mkdir 1-person
cd 1-person
mkdir 01
mkdir 02
cd ..
mv 01.zip 1-person/01
mv 02.zip 1-person/02
cd 1-person/01/
python3 unzip.py 01.zip
rm 01.zip
cd ../02/
python3 unzip.py 02.zip
rm 02.zip
cd ../../
rm 1-person.zip

mkdir 2-person
mv 2-person.zip 2-person
cd 2-person
python3 unzip.py 2-person.zip
rm 2-person.zip
cd ..


mkdir 3-person
mv 3-person.zip 3-person
cd 3-person
python3 unzip.py 3-person.zip
rm 3-person.zip
cd ..


python3 unzip.py multi-persons.zip
mkdir 4-person
mv 4-person.zip 4-person/
mkdir 5-person
mv 5-person.zip 5-person/
rm multi-persons.zip

cd 4-person
python3 unzip.py 4-person.zip
rm 4-person.zip
cd ..

cd 5-person
python3 unzip.py 5-person.zip
rm 5-person.zip
cd ..
