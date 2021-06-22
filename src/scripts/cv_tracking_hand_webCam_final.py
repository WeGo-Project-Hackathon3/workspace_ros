#! /home/ssac27/anaconda3/envs/hack3/bin/python

import time
import os

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

# bounding box utils
from bboxes_utils import * 
import core.utils as utils

# hand
from hand_webCam import hand_detection

# ROS
import rospy
from geometry_msgs.msg import Twist
from ros_pub import *

# depth camera 
import pyrealsense2 as rs
from realsense_depth import *
from depth_cam_distance import depth_action


# multiprocessing 
from functools import partial
import multiprocessing
from multiprocessing import Pool, Process

absolute_path = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(absolute_path, 'checkpoints/yolov4-tiny-416-small')
IOU_THRESHOLD = 0.3
SCORE_THRESHOLD = 0.35
INPUT_SIZE = 416


# load model
saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
print("MODEL_PATH : ", MODEL_PATH)

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

def main(video_path,pub):
    # processing
    num_cores = multiprocessing.cpu_count()
    pool = Pool(num_cores)

    # Definition of the parameters

    iou_ =0.0
    algo = "csrt"  # 최소 8 ~ 21 
    #algo = "csrt" # kcf보다 느리지만 정확도 증가 
    success = False
    #DETECT_AFTER = 100 # 초기 frame 반복당 tracker initialize
    DETECT_AFTER = 35
    frame_number = -1
    pre_frame_chk = frame_number

    # 초기 x,y,w,h 
    T_W, T_H = 80, 80
    x,y,w,h = 290,190,T_W,T_H


    # ============ cap read ============
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    #dc = DepthCamera() 
    #ret, _, frame = dc.get_frame()
    
    (H, W) = frame.shape[:2]

    # Bounding Box initial 
    initBB = ( x, y, w, h)
    hand_initBB = (0,0,0,0)
    pre_initBB = (0,0,0,0)
    
    # OpenCV Traking API 
    tracker = OPENCV_OBJECT_TRACKERS[algo]()

    # hand variable
    action = ''
    hand_center = (0)
    init_chk = True    # 처음 시작 부분 check
    init_first = True  # hand 부분만 적용

    # ros
    # 차량에 publish 할때의 값
    angular_z_pre = 0.0
    angular_z = 0.0
    linear_x_pre = 0.0
    linear_x = 0.0
     
    occlusion = False
    DIST_AFTER = 2

    # PUB_AFTER 만큼 반복 후 발행
    PUB_AFTER = 1
    twist = Twist()



    while cap.isOpened()  :
        frame_number+=1
        #ret, depth_frame, frame = dc.get_frame()
        ret, frame = cap.read() # joo.jg 주석 
        if not ret:
            break
        

        # cam의 화면과 손의 위치가 동일하게 뒤집어줌
        frame = cv2.flip(frame, 1)
        image_np = np.array(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_input = cv2.resize(image_np, (INPUT_SIZE, INPUT_SIZE))
        frame_input = frame_input / 255.
        frame_input = frame_input[np.newaxis, ...].astype(np.float32)
        frame_input = tf.constant(frame_input)
        start_time = time.time()

        # model에 frame input을 넣어서 예측
        pred_bbox = infer(frame_input)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # tf.image.combined_non_max_suppression 
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=35,
            max_total_size=35,
            iou_threshold=IOU_THRESHOLD,
            score_threshold=SCORE_THRESHOLD
        )


        # 데이터를 numpy 요소로 변환 및 사용하지 않는 요소는 잘라낸다. 
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)


        # ========== person detection end ===================
        # ================ hand detection start ===============

        img = frame.copy()
        
        if init_first and num_objects ==1:

            #  move or stop  동작을 받아야 다음 step 가능

            if action == '' :
                x_min, y_min = bboxes[0][:2]
                #y_min = int(max(y_min-20, 0)); x_min = int(max(x_min-10, 0))
                action, (hand_x, hand_y, hand_w, hand_h) = hand_detection(cap, x_min, y_min)

            hand_center = (hand_x + hand_w//2)


        # =============== hand detection end =================
            # tracker 초기화
            # get_coordinates(bbox, 해당 bbox 안에 추적을 할 bounding_box)
            # bboxes : x_min, y_min, x_max, y_max  [사람의 bounding box]
            
            hand_initBB, trueBB,iou_ = get_coordinates(bboxes, x_min, y_min, (hand_x + hand_w) , ( hand_y+hand_h) )
            hand_initBB = tuple(map( int, hand_initBB) )
            
            # 시작 hand bbox를 initBB로 초기화 해준다.
            pre_initBB = (  hand_initBB[0] + hand_initBB[2]//2  ,  hand_initBB[1] + hand_initBB[3]//2 ) #,  hand_initBB[2],  hand_initBB[3] )
            
            tracker.init(frame, hand_initBB)
            
            (success, box) = tracker.update(frame)
            if success :
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 120), 2)
                cv2.putText(frame, f"hand tracking", (x + w//2, y+h +10), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 120, 120), 2)
                pred_bbox = [bboxes, scores, classes, num_objects]
                result = utils.draw_bbox(frame, pred_bbox,pub)

            init_first = False


        # 100 frame 단위로 DETECT trackes 확인  DETEVTER
        if not init_first and frame_number % DETECT_AFTER == (DETECT_AFTER-1) or not success  :
            if num_objects>=1 :

                new_bboxes = get_box_distance(bboxes, [x, y, x+w, y+h],  W)

                # ==========================   process test _coordi (start)  =========================

                func = partial(get_coordinates, new_bboxes, x, y, x+w )
                temp = pool.map(func , [y+h] )
                initBB, trueBB,iou_  = temp[-1]

                # ==========================   process test _coordi (end)  =========================

                initBB = tuple(map( int, initBB) )
                trueBB = tuple(map( int, trueBB) )


                # ==== 06.22 ===========
                # initBB  수정 전 rectangle
                cv2.putText(img, f"iou_scores : {iou_} "  ,  (initBB[0] , initBB[1] )  , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                # 기존 tracker initBOX와 사람의 bbox를 기준으로 한 initBB 비교가 필요한 경우 아래 주석 해제  
                #cv2.rectangle(img, (initBB[0], initBB[1]), (initBB[0] + initBB[2], initBB[1] + initBB[3]),(255, 255, 0), 2)
                #cv2.putText(img, f"old initBB box : {initBB} ",  ( (initBB[0]  ) , ( initBB[1] -20 ) ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (102, 160, 145), 2)
                cv2.rectangle(img, (trueBB[0], trueBB[1]), (trueBB[0] + trueBB[2], trueBB[1] + trueBB[3]),(255, 255, 0), 2)


                init_hand_coorect = -1* ( W//2 - hand_center )//2

                if init_chk:
                    initBB =  ( trueBB[0] + (trueBB[2]//2) - T_W//2  + init_hand_coorect ), ( trueBB[1] + (trueBB[3]//2) - T_H//2)  , T_W, T_H #, initBB[2], initBB[3]  
                else : 
                    initBB =  ( trueBB[0] + (trueBB[2]//2) - T_W//2  ), ( trueBB[1] + (trueBB[3]//2) - T_H//2)  , initBB[2], initBB[3]#, T_W, T_H   


                print(f"new initBB :", initBB)
                cv2.rectangle(img, (initBB[0], initBB[1]), (initBB[0] + initBB[2], initBB[1] + initBB[3]),(147, 200, 100), 2)
                cv2.putText(img, f"new initBB box : {initBB} ",  ( (initBB[0] + initBB[2] - 30) , ( initBB[1] + initBB[3]+20 ) ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (102, 160, 145), 2)
                cv2.imshow('yolo', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                tracker = OPENCV_OBJECT_TRACKERS[algo]()

                tracker.init(frame, initBB)

        if not init_first :

            (success, box) = tracker.update(frame)

            
            depth_dist = 0
            
            if  success  :
                (x, y, w, h) = [int(v) for v in box]
                #x = min(W,x+10); y = max(0,y-10)
                cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 120), 2)
                
                if len(box) <1 :
                    print(" 사람이 없습니다. ")
                
                # update 하기 위한 initBB 
                diff_dist =  round( np.linalg.norm( ((pre_initBB[0]) /W) - ((x+ w//2)/W) ),2)
                if diff_dist > 0.23 :#and DIST_AFTER:
                    print("track bounding box 위치 문제 발생") 
                    x = pre_initBB[0]
                    y = pre_initBB[1] 


                # depth로 범위 안에 포함되는지 와 거리 가져온다.
                cv2.putText(frame, f"current :  ", (x + w//2, y+h +10), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 120, 120), 2)

                pred_bbox = [bboxes, scores, classes, num_objects]
                result = utils.draw_bbox(frame, pred_bbox)

                # ==========================   process pool _move (start)  =========================
                
                func = partial(get_move, W//2,  x+(w//2), hand_center )
                temp = pool.map(func , [init_chk] )
                twist,angular_z, linear_x = temp[-1]
                
                # ==========================   process pool _move (end)  =========================
                init_chk= False
            
            ln_bboxes = len(bboxes) 
            # depth cam으로 쟀을때 거리가 만족되는 경우 


            # =================== 21.06.19 수정 =============
            if ln_bboxes >=1 :
                pre_frame_chk = frame_number

                if abs(angular_z_pre) - abs(angular_z) > 0.3 :#
                    print(f"occlusion 발생  :  {occlusion }")
                    occlusion = not occlusion
                    # ----- 2021 06 - 19 코드 수정 ----
                    time_wait_pub(0.01, pub,linear= 0, angular=  angular_z_pre/2)
                    occlusion = False
                elif not occlusion :#and ok:
                    print(" not occlusion, 정상 publish")
                    # 매 frame마다 발행이 아닌 일정 반복마다 발행
                    if frame_number % PUB_AFTER == (PUB_AFTER-1) : 
                        pub.publish(twist)


            elif ln_bboxes < 1 :
                # 사람이 없는 경우 stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                pub.publish(twist)

            occlusion = False

            angular_z_pre = angular_z
            #linear_x_pre = linear_x
            pre_initBB = (initBB[0]+ initBB[2]//2 , initBB[1] + initBB[3]//2 ) # , initBB[2], initBB[3]) 
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)


        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('result', result)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            #dc.release()
            cap.release()
            pool.close()
            pool.join()
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    rospy.init_node('scout_ros_detector_test', anonymous=False)
    pub = rospy.Publisher('/cmd_vel', 
        Twist, 
        queue_size=10)
    video_path = -1
    main(video_path,pub)

