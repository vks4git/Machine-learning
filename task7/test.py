from kmeans import K_means
from dbscan import DBScan
from idx2numpy import convert_from_file
import numpy
from scipy.spatial.distance import hamming

images = numpy.reshape(convert_from_file("train-images.idx3-ubyte"), (60000, 784)).astype("float64")
labels = convert_from_file("train-labels.idx1-ubyte")
images = numpy.multiply(images, 1 / 255)

# kmeans = K_means(10, 1)
# kmeans.fit(images)
# print(kmeans.score(labels))

dbscan = DBScan(7, 78)
dbscan.fit(numpy.array([images[i] for i in range(1000)]))
print(dbscan.score(numpy.array([labels[i] for i in range(1000)])))
print(dbscan.clusters())
