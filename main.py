import cv2

#img=cv2.imread("img_6.png")
pressHold=0.45
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)



names=[]

cocoNames="coco.names"

with open(cocoNames,"rt") as reader :
    names=reader.read().rstrip("\n").split("\n")

print(names)

frozenPath="frozen_inference_graph.pb"

ssdPath="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

net=cv2.dnn_DetectionModel(frozenPath,ssdPath)

net.setInputSize(320,320)

net.setInputScale(1.0/127.5)

net.setInputMean((127.5,127.5,127.5))

net.setInputSwapRB(True)


while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=pressHold)
    print(classIds,bbox)

#print(classIds)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
             cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
             cv2.putText(img, names[classId - 1].upper(), (box[0] + 10, box[1] + 30),
             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
             cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("window",img)

    cv2.waitKey(1)



