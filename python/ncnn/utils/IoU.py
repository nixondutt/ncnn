import re
import numpy as np
import pathlib
import yaml

def process_color_box_one(filepath):
    path =  pathlib.Path.cwd().joinpath(filepath)
    if path.exists():
        with open(filepath) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            diction = yaml.load(file, Loader=yaml.FullLoader)
            return diction
    else:
        print("The file path does not exists")
        return
    

#Selected box
def process_selected_box_two(filepath):
    box = {}
    path = pathlib.Path.cwd().joinpath(filepath)
    if path.exists():
        with open(filepath) as file:
            diction = yaml.load(file, Loader = yaml.FullLoader)
        for key,value in diction.items():
            (val1,val2) ,(val3,val4) = value
            box[key] = [val1,val2,val3,val4]
        return box
    else:
        print("The file does not exist.")
        return  

#returns the iou of two bounding boxes
def iou(box1, box2):
    x1,y1,x2,y2 = box1
    x11,y11,x22,y22 = box2
    maxX1 = np.maximum(x1,x2)
    maxY1 = np.maximum(y1,y2)
    maxX11 = np.maximum(x11,x22)
    maxY11 = np.maximum(y11,y22)
    maxX = np.maximum(maxX1,maxX11)
    maxY = np.maximum(maxY1, maxY11)
    if maxX == maxX1:
        minX = np.minimum(x1,x2)
        if minX > maxX11:
            return 0
    else:
        minX = np.minimum(x11,x22)
        if minX > maxX1:
            return 0
    
    if maxY == maxY1:
        minY = np.minimum(y1,y2)
        if minY > maxY11:
            return 0
    else:
        minY = np.minimum(y11,y22)
        if minY > maxY1:
            return 0        
    
    xi1 = np.maximum(box1[0], box2[0])  # for lower point index of x  
    yi1 = np.maximum(box1[1], box2[1])  # for lower index of y
    xi2 = np.minimum(box1[2], box2[2])  # for higher index of x 
    yi2 = np.minimum(box1[3], box2[3])  # for higher index of y
    inter_area = np.abs((xi1 - xi2) * (yi1 - yi2))  # with respect to above index points get area of overlapping boxes 
    #print("co-ord of overlapping area  {} ,{} ,{} ,{} \n  inter_area :{}" .format(xi1 ,yi1 ,xi2 ,yi2 , inter_area) )

    box1_area = np.abs((box1[2] - box1[0]) * (box1[3] - box1[1])) # coordinates geometry (x2-x1) *(y2-y1) for box areas 
    box2_area = np.abs((box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area  # total box's areas - overlapping area for union  
    #print('boxes_area without overlapping area' , box1_area ,box2_area ,'union_area' ,union_area)

    iou = inter_area / union_area 
    return iou



if __name__ == '__main__':
    bb_twos = process_selected_box_two()   #rectangle box
    cd_ones_dict = process_color_box_one() #four cornered box
    bb_threes = process_detected_box_three()
    for ID,box1 in bb_twos.items():
        check = True
        for box2 in bb_threes:
            if check == True:
                IOU = iou(box1,box2)
                if IOU > 0.30:
                    check = False
                    print("Space number {} is filled and IOU is {}".format(ID,IOU))

