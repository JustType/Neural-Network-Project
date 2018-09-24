import cv2




cap = cv2.VideoCapture(0)

ret, frame = cap.read()
print ret
print frame.shape
#frame = frame.transpose(2,0,1)
print frame.shape

#cv2.imwrite('cap.jpg', frame)


#quit()
counter = 0

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
        out = cv2.imwrite('gallery/5/capture' + str(counter) + '.jpg', neural_frame)
        counter += 1
        #break

cap.release()
cv2.destroyAllWindows()
