# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import sys
import cv2
import time
import numpy as np
import ncnn
from ncnn.model_zoo import get_model
# from ncnn.utils import draw_detection_objects
import time
import yaml
from ncnn.utils import IoU

# Loading data and box cordinates files

parking_box_label = sys.argv[2]
with open(parking_box_label) as file:
    parsed_yaml_file = yaml.load(file, Loader = yaml.FullLoader)
four_corners = parsed_yaml_file['four_corners']
rectangle_cordinates = parsed_yaml_file['rectangle_cordinates']
cd_ones_dict = IoU.process_color_box_one(four_corners)
bb_twos = IoU.process_selected_box_two(rectangle_cordinates)



def draw_detection_objects(image, class_names, objects, min_prob=0.0):
    for ID, cordinates in cd_ones_dict.items():
        image = cv2.line(image,cordinates[0], cordinates[1],(247,174,7),2)
        image = cv2.line(image,cordinates[1], cordinates[2],(247,174,7),2)
        image = cv2.line(image,cordinates[2], cordinates[3],(247,174,7),2)
        image = cv2.line(image,cordinates[3], cordinates[0],(247,174,7),2)
        midx, midy = cordinates[4][0], cordinates[4][1]
        image = cv2.putText(image, str(ID), (midx,midy), cv2.FONT_HERSHEY_SIMPLEX,.5, (7,247,57),1,cv2.LINE_AA)
    idDict = {} 
    for obj in objects:
        if int(obj.label)!= 2 or obj.prob < min_prob:
            continue
        framebox = [int(obj.rect.x), int(obj.rect.y), int(obj.rect.x + obj.rect.w), int(obj.rect.y + obj.rect.h)]
        for ID, box1 in bb_twos.items():
            # if lot is already taken, it will not be check for the next car
            if ID not in idDict: 
                check = True
                if check: # if check true, box1 is not checked yet
                    IOU = IoU.iou(box1, framebox)
                    if IOU > 0.40:
                        check = False
                        cd_box = cd_ones_dict[ID]
                        for cord in range(0,len(cd_box)-2):
                            image = cv2.line(image,cd_box[cord],cd_box[cord+1],(7,7,247),2)
                        image = cv2.line(image,cd_box[cord+1],cd_box[0],(7,7,247),2)  #for forth line
                        midx, midy = cd_box[cord+2][0], cd_box[cord+2][1]                          # cordinates of text   Code added at 2/8/2021
                        image = cv2.putText(image, str(cd_box[cord+2][2]), (midx,midy), cv2.FONT_HERSHEY_SIMPLEX,.5, (7,7,247),1,cv2.LINE_AA)
                        idDict[ID] = True

                else:
                    break # going to check for the next car

        # print(
        #     "%d = %.5f at %.2f %.2f %.2f x %.2f\n"
        #     % (obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.w, obj.rect.h)
        # )

        cv2.rectangle(
            image,
            (int(obj.rect.x), int(obj.rect.y)),
            (int(obj.rect.x + obj.rect.w), int(obj.rect.y + obj.rect.h)),
            (255, 0, 0),
        )
        

        text = "%s %.1f%%" % (class_names[int(obj.label)], obj.prob * 100)

        label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        x = obj.rect.x
        y = obj.rect.y - label_size[1] - baseLine
        if y < 0:
            y = 0
        if x + label_size[0] > image.shape[1]:
            x = image.shape[1] - label_size[0]

        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + label_size[0]), int(y + label_size[1] + baseLine)),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            image,
            text,
            (int(x), int(y + label_size[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )

    return image

    # cv2.imshow("image", image)
    # cv2.waitKey(0)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: %s [imagepath]\n" % (sys.argv[0]))
        sys.exit(0)

    imagepath = sys.argv[1]
    cap = cv2.VideoCapture(imagepath)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('../data/output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'),30,(frame_width,frame_height))
    net = get_model(
    "yolov5s",
    target_size=640,
    prob_threshold=0.25,
    nms_threshold=0.45,
    num_threads=4,
    use_gpu=False,
)
    
    while cap.isOpened():
        
        # m = cv2.imread(imagepath)
        ret, m = cap.read()
        if m is None:
            print("cv2.imread %s failed\n" % (imagepath))
            sys.exit(0)


        start_time = time.time()
        objects = net(m)
        fps_obj = 1.0/(time.time() - start_time)
        image = draw_detection_objects(m, net.class_names, objects)
        fps = 1.0/(time.time() - start_time)
        print('FPS for park allot + obj det:  %.2f and FPS for obj_det:   %.2f'%(fps,fps_obj))
        cv2.putText(
            image,
            'FPS : %.2f'%fps_obj,
            (120, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (210, 20, 85),
            1,
            cv2.LINE_AA
        )
        out.write(image)
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


