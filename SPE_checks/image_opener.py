from PIL import Image

plot=input('Image to be opened?: ')
path=home/awalsh272/SPE/data
fullpath=path+'/'+plot

image= Image.open(fullpath)
image.show()
