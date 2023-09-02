
from ldm.data.imagenet import ImageNetTrain, ImageNetValidation
print('Downloading ImageNetTrain')
train = ImageNetTrain()

print('Downloading ImageNetValidation')
for k, v in train[0].items():
    print(k, v.shape if hasattr(v, 'shape') else type(v))
val = ImageNetValidation()

print('finished')
