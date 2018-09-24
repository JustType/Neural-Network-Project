## Place your hand into frame and press 'Q'
import cv2
import cnn
from time import sleep
import torch


model = cnn.CNN(1,512,6)
cap = cv2.VideoCapture(0)
model.load_state_dict(cnn.load_model())
modif = 50
neural_frame = None
while(True):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    rgb[0:500, 100] = 0
    rgb[0:500, 500] = 0
    rgb[500, 100:500] = 0

    neural_frame = frame[0:500, 100:500]
    cv2.imshow('Partial Frame', neural_frame)

    #cv2.imshow('frame', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('temp.jpg', neural_frame)
        image = cv2.imread('temp.jpg')
        t_img = cnn.preprcess_image(image)
        #print t_img.size()
        with torch.no_grad():
            predict = model(t_img)

        #print predict
        print torch.max(predict, 1)[1].item()



cap.release()
cv2.destroyAllWindows()
