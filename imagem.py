from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

def getDataImage( path):
    img = Image.open( path )
    pixels = img.load() 
    data = []
    pixel = []
    for i in range( img.size[0]):
        for j in range( img.size[1] ):
             pixel = pixels[i,j]
             data.append( pixel[0] )
             data.append( pixel[1] )
             data.append( pixel[2] )

    exif_data = img._getexif()
    exif_data
    return data

size = 50 * 50 * 3

network = buildNetwork( size, 100, 30, 1 )
dataSet = SupervisedDataSet( size, 1 )

dataSet.addSample ( getDataImage( 'img\\nl-1.png' ), 0 )
dataSet.addSample ( getDataImage( 'img\\nl-2.jpg' ), 0 )
dataSet.addSample ( getDataImage( 'img\\nl-3.png' ), 0 )
dataSet.addSample ( getDataImage( 'img\\nl-4.jpg' ), 0 )
dataSet.addSample ( getDataImage( 'img\\l-1.jpg' ), 1 )
dataSet.addSample ( getDataImage( 'img\\l-2.jpg' ), 1 )
dataSet.addSample ( getDataImage( 'img\\l-3.jpg' ), 1 )
dataSet.addSample ( getDataImage( 'img\\l-4.jpg' ), 1 )

trainer = BackpropTrainer( network, dataSet)
error = 1
iteration = 0
outputs = []
file = open("outputs.txt", "w")
while error > 0.001:
    error = trainer.train()
    outputs.append( error )
    file.write( str(error)+"\n" )
    iteration += 1
    print ( iteration, error )


plt.plot( outputs )
plt.xlabel('Iteracoes')
plt.ylabel('Erro Quadratico')

plt.savefig("grafico.png", bbox_inches='tight')

plt.show()


name = ['lt-1.jpg', 'lt-2.jpg', 'nlt-1.jpg', 'nlt-2.jpg']
for i in range( len(name) ):
    path = "img\\test\\" + name[i]
    res = network.activate( getDataImage( path ) )
    file.write(str(path)+"\n"+str(res)+"\n")
    print ( path )
    print ( res )
    
file.close()