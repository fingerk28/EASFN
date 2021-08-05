from zipfile import ZipFile
from tqdm import tqdm
import sys

with ZipFile(sys.argv[1], 'r') as zipf:
	for name in tqdm(zipf.namelist(), desc=sys.argv[1], unit='files'):
			zipf.extract(name)
	zipf.close()