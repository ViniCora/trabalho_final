import cv2
import numpy as np

proto_txt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'

img_path = 'rubik.jpg'

net = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
#Usando assim por causa do espaço de cor LAB
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

bw_image = cv2.imread(img_path)
normalized = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

#Model treinado em 224x224

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
AB = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(AB, (bw_image.shape[1], bw_image.shape[0]))
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = (colorized * 255.0).astype("uint8")

cv2.imshow("BW IMAGE", bw_image)

cv2.imshow("Colorized IMAGE", colorized)

cv2.waitKey(0)
cv2.destroyAllWindows()
