import os
import random
from shutil import copyfile

paths = []

for files in os.walk("/Users/nikhilpunwaney/Desktop/fbank_by_utt"):
	for sub in files[2]:
		paths.append((files[0] + '/' + sub, sub))

paths = paths[1:]

testPath = "/Users/nikhilpunwaney/Desktop/test/"
trainPath = "/Users/nikhilpunwaney/Desktop/train/"

for f in paths:
	filePath = f[0]
	fileName = f[1]
	if random.random() <= 0.09:
		copyfile(filePath, trainPath + fileName)
	else:
		copyfile(filePath, testPath + fileName)


