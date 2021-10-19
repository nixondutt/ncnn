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
bb_twos = IoU.process_selected_box(rectangle_cordinates)



def draw_detection_objects(image, class_names, objects, min_prob=0.0):
    for ID, cordinates in cd_ones_dict.items():
    im0s = cv2.line(im0s,cordinates[0], cordinates[1],(247,174,7),2)
    im0s = cv2.line(im0s,cordinates[1], cordinates[2],(247,174,7),2)
    im0s = cv2.line(im0s,cordinates[2], cordinates[3],(247,174,7),2)
    im0s = cv2.line(im0s,cordinates[3], cordinates[0],(247,174,7),2)
    midx, midy = cordinates[4][0], cordinates[4][1]
    im0s = cv2.putText(im0s, str(ID), (midx,midy), cv2.FONT_HERSHEY_SIMPLEX,.5, (7,247,57),1,cv2.LINE_AA) 
    for obj in objects:
        if obj.prob < min_prob:
            continue

        print(
            "%d = %.5f at %.2f %.2f %.2f x %.2f\n"
            % (obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.w, obj.rect.h)
        )

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
    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n" % (sys.argv[0]))
        sys.exit(0)

    imagepath = sys.argv[1]
    cap = cv2.VideoCapture(imagepath)
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

        image = draw_detection_objects(m, net.class_names, objects)
        fps = 1.0/(time.time() - start_time)
        print(f'FPS:  :{fps}')
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

