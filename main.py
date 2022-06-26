import dlr
import numpy as np
import time
import cv2
import postprocess as pp


model = dlr.DLRModel('./tiny_yolov3_model', 'cpu', 0)

f = open("coco.names", 'r')
coco = f.readlines()
print(coco)

cap = cv2.VideoCapture("qtiqmmfsrc name=qmmf ! video/x-raw, format=NV12, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)

while(cap.isOpened()):
    ret, img = cap.read()
    orig_img = img.copy()
    img = cv2.resize(img, (416, 416))
    img = img/255.
    img = np.reshape(img, (1, 416, 416, 3))
    img = np.moveaxis(img, -1, 1)

    s = time.time()
    y = model.run(img)
    e=time.time()
    
    anchors = np.reshape(y[2], (6,2))
    out_26x26 = np.reshape(np.moveaxis(y[0], 1, -1), (1,26,26,3,85))
    out_13x13 = np.reshape(np.moveaxis(y[4], 1, -1),(1,13,13,3,85))

    orig_box, orig_score, orig_class = pp.postprocess_box(out_26x26, anchors, y[1], (26,26))
    orig13_box, orig13_score, orig13_class = pp.postprocess_box(out_13x13, anchors, y[5], (13,13))

    bboxes = np.concatenate([orig_box, orig13_box], axis=1)
    scores = np.concatenate([orig_score, orig13_score], axis=1)
    classes = np.concatenate([orig_class, orig13_class], axis=1)

    boxes, classes, scores = pp.filter_boxes(bboxes, scores, classes)
    boxes_len = boxes.shape[0]
    for i in range(boxes_len):
        x1, y1, x2, y2 = (boxes[i]/416.0)
        x1 = int(x1*orig_img.shape[1])
        x2 = int(x2*orig_img.shape[1])
        y1 = int(y1*orig_img.shape[0])
        y2 = int(y2*orig_img.shape[0])

        orig_img = cv2.rectangle(orig_img, (x1,y1), (x2, y2), (255,255,0), 3)
        orig_img = cv2.putText(orig_img, coco[classes[i]], (x1, y1-20), 1, 1, (255,255,0), 1, 1)
    cv2.imwrite("out.jpg", orig_img)
    print("FPS:", 1/(e-s))

