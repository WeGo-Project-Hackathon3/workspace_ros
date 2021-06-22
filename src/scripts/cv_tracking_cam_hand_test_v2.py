#! /home/ssac27/anaconda3/envs/hack3/bin/python

import time
import os
import tensorflow as tf
import numpy as np
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from core.config import cfg
import matplotlib.pyplot as plt
from hand_test2 import hand_detection

# ROS
import rospy
from geometry_msgs.msg import Twist

absolute_path = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(absolute_path, 'checkpoints/yolov4-tiny-416')
IOU_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.3
INPUT_SIZE = 416
algo = "kcf"

success = True
DETECT_AFTER = 30
x,y,w,h = 0,0,0,0

# hand_detection 고려해야함
init_check = True

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

def get_iou(boxA, boxB):
	""" Find iou of detection and tracking boxes
	"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)

	# return the intersection over union value
	return iou

def get_coordinates(box, x, y, x1, y1):
	""" Get co-ordinates of flaged person
	"""
	if len(box) == 0:
#		print('!!!!!!!!No person detected!!!!')
		return
	iou_scores = []
	for i in range(len(box)):
		iou_scores.append(get_iou(box[i],[x,y,x1,y1]))

	index = np.argmax(iou_scores)
	print(iou_scores, ' ',box, ' ', x, y, x1, y1)

	if np.sum(iou_scores) == 0:
		# print('#'*20, 'No Match found', '#'*20)
		box = np.array(box)
		distance = np.power(((x+x1)/2 - np.array(box[:,0] + box[:,2])/2),2) + np.power(((y+y1)/2 - (box[:,1]+box[:,3])/2), 2)
		index = np.argmin(distance)

	x, y, w, h = box[index][0], box[index][1], (box[index][2]-box[index][0]), (box[index][3]-box[index][1])
	initBB = (x+w//2-60,y+h//2-45,120,120)
    #initBB = (x+w//2-60,y+h//2-45,100,100)
    #initBB = (x+w//2-50,y+h//2-50,100,100)  # default  iou box
    

	return initBB, (x,y,x+w,y+h), iou_scores

# ================ 정해진 사람만 구하기 =================
# 센터 값을 기준으로 두 점사이의 거리를 구한다.
def get_box_dist(boxA, boxB, W,  H=0, isWidth=True):
    # W 거리는 보류한다.
    boxA_cx = (boxA[0] + boxA[2])//2
    boxA_cy = (boxA[1] + boxA[3])//2
    boxA_c = np.array( ( boxA_cx, boxA_cy) )

    boxB_cx = (boxB[0] + boxB[2])//2
    boxB_cy = (boxB[1] + boxB[3])//2
    boxB_c = np.array( (boxB_cx, boxB_cy ) )

    if isWidth :
        # x 좌표만 사용해서 거리 비교
        dist = round( np.linalg.norm(boxA_cx-boxB_cx),2 )
        return round(dist/W ,4)
    else :
        # 두 점사이의 거리
        dist = round( np.linalg.norm(boxA_c-boxB_c),2 )
        return round(dist/W ,4)



# track_bbox, 전체 bbox를 받은 후 가장 거리가 덜 튕기는 box를 구한다.
def get_distance(bbox, track_box,  W, H=0):
    if len(bbox) == 0: # no person
        return
    select_box = []
    bias_ratio = 0.2
    x,y,x1,y1 = track_box[0], track_box[1], track_box[0]+track_box[2], track_box[1]+track_box[3]  
    for i in range(len(bbox)):
        ratio =  get_box_dist(bbox[i], [x,y,x1,y1], W, H=0,isWidth=False )
        if ratio < bias_ratio :
            select_box.append(bbox[i])

    return select_box


def get_move(W,track_w, H=0, track_H=0):
    twist = Twist()
    x_bias = 0.1
    z_bias = 0.2
    angular_z = round(-1.0 * (W-track_w)/W,5  ) 
    angular_chk = 1  #  0 : left  ,   1  : right
    if angular_z < 0 :  # 우회전 해야함 
        angular_chk = -1
    
    #if angular_z < 0: 
    
    angular_z += (z_bias*angular_chk)
    # 최소값 정의 angular
    if abs(angular_z) <= 0.25+ abs(z_bias) :
        angular_z = 0.0


    # 최대 값 정의 angular 
    if abs(angular_z) > 0.8 :
        angular_z = 0.8 * angular_chk
    #elif angular_z < -0.8:
    #    angular_z = -0.8

    

    # angular에 값이 들어가 있는 경우
    # angular에 절반 만큼의 속도를 부여
    linear_x = abs(angular_z/2) 
    if  abs(angular_z) > 0: 
        twist.linear.x = linear_x + x_bias
    else : 
        twist.linear.x = 0.25 + x_bias 
    print("angular_z : ", angular_z/2) # cam 상에서 내 위치 : - right   + left   실제 적용해야하는 값 : +  right,    - left  (화면이 반대이므로)
    print("linear_x : ", twist.linear.x) # - right   + left서
    
    twist.linear.y = 0
    twist.linear.z = 0

    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = angular_z/2
    #if abs(angular_z) <= 0.2 
    #    twist.angular.z = 0.0


    return twist, angular_z


def init_twist():
    twist = Twist()
    twist.linear.x = 0
    twist.linear.y = 0
    twist.linear.z = 0

    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = 0
    return twist

def time_wait_pub(duration, pub,linear=0.2, angular=0.3):
    twist = Twist()
    twist.linear.x = linear; twist.linear.y = 0; twist.linear.z = 0

    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = angular
    time_chk = time.time()
    while time.time() - time_chk < duration : 
        pub.publish(twist)

def time_wait(duration):
    time_chk = time.time()
    #while time.time() - time_chk < duration :
    time.sleep(duration)


# =============================== hand option에 따라 움직임을 재어할때 사용 ===============================
def hand_option(action,pub) :
    twist = Twist()

    if action == 'move':
        twist.linear.x = 0.2; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0

    elif action == "stop" :
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        
    pub.publish(twist)

def main(video_path,pub):
    # Definition of the parameters
    
    # Tracking
    init_first = True

    frame_number = -1
    iou_ =0.0
    algo = "kcf" 

    success = False
    #DETECT_AFTER = 100 # 초기 frame 반복당 tracker initialize
    DETECT_AFTER = 30
    x,y,w,h = 175,275,120,120


    # hand variable
    action = ''

    # ros
    angular_z_pre = 0.0
    angular_z = 0.0
    occlusion = False
    twist = Twist()

    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    # 초기 Bounding Box 정의
    init_BB = ( x, y, 120, 120)
    hand_initBB = (0,0,0,0)
    (H, W) = frame.shape[:2]
    tracker = OPENCV_OBJECT_TRACKERS[algo]()




    r = rospy.Rate(30)

    while cap.isOpened():
        frame_number+=1
        ret, frame = cap.read()
        if not ret:
            break
        
        # cam의 화면과 손의 위치가 동일하게 뒤집어줌
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_input = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
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
            max_output_size_per_class=50,
            max_total_size=50,
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
        
        img = frame.copy()


        # ========== person detection end ===================
        # ================ hand detection start ===============
        if init_first and num_objects ==1:
            # cap
            #action= ''
            #  move or stop  동작을 받아야 다음 step 가능

            if action == '' :
                x_min, y_min = bboxes[0][:2]
                action, (hand_x, hand_y, hand_w, hand_h) = hand_detection(cap, x_min, y_min)
                #action, (x, y, w, h) = hand_detection(cap)



        # =============== hand detection end =================
            # tracker 초기화
            # get_coordinates(bbox, 해당 bbox 안에 추적을 할 bounding_box)
            # bboxes : x_min, y_min, x_max, y_max  [사람의 bounding box]
            #

            hand_initBB, trueBB,iou_ = get_coordinates(bboxes, x_min, y_min, hand_x + hand_w, hand_y+hand_h)
            #hand_initBB, trueBB,iou_ = get_coordinates(bboxes, x, y, x+w, y+h) 
            hand_initBB = tuple(map( int, hand_initBB) )
            tracker = OPENCV_OBJECT_TRACKERS[algo]()
            tracker.init(frame, hand_initBB)

            (success, box) = tracker.update(frame)
            if success :
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 120), 2)
                cv2.putText(frame, f"hand tracking", (x + w//2, y+h +10), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 120, 120), 2)
                pred_bbox = [bboxes, scores, classes, num_objects]
                result = utils.draw_bbox(frame, pred_bbox,pub)

            init_first = False

        # ================== new hand detection move check ===================
        # ================   hand를 while 내에서 계속 돌려볼 시 아래 사용 (1) ==========
        '''
        print("num_objects : ", num_objects)
        DETECTER_check = 2*DETECT_AFTER
        if not init_first and num_objects >= 1 and frame_number % DETECTER_check == (DETECTER_check-1):
            if action != '' :
                x_min, y_min = bboxes[0][:2]
                action, (hand_x, hand_y, hand_w, hand_h) = hand_detection(cap, x_min, y_min)
                hand_option(action,pub)
        '''
        
        # 지정한 {DETECT_AFTER} frame 단위로 DETECT trackes 확인

        if not init_first and frame_number % DETECT_AFTER == (DETECT_AFTER-1) or not success :
            if num_objects>=1 :
                bboxes = get_distance(bboxes, x, y, x+w, y+h,  W)
                initBB, trueBB,iou_ = get_coordinates(bboxes, x, y, x+w, y+h)

                initBB = tuple(map( int, initBB) )


                cv2.putText(img, f"iou_scores : {iou_} ",  (initBB[0] , initBB[1] )  , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (initBB[0], initBB[1]), (initBB[0] + initBB[2], initBB[1] + initBB[3]),(255, 255, 0), 2)


                cv2.rectangle(img, (trueBB[0], trueBB[1]), (trueBB[0] + trueBB[2], trueBB[1] + trueBB[3]),(255, 255, 0), 2)
                cv2.putText(img, f"trueBB : {iou_} ",  (trueBB[2], trueBB[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                cv2.imshow('yolo', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                initBB =  ( trueBB[0] + (trueBB[2]//2) - (trueBB[2]//4) + (init_BB[2]//2) ) , ( trueBB[1] + (trueBB[3]//2) - (trueBB[3]//4) + (init_BB[3]//2)  )  , initBB[2], initBB[3]  

                tracker = OPENCV_OBJECT_TRACKERS[algo]()
                # more tracking
                #tracker.init(frame, trueBB)
                print("initBB : ", initBB)
                tracker.init(frame, initBB)


        
        
        (success, box) = tracker.update(frame)
        print("box : ", box)

        
        if success :
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 120), 2)
            cv2.putText(frame, f"current tracking", (x + w//2, y+h +10), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 120, 120), 2)
            pred_bbox = [bboxes, scores, classes, num_objects]
            result = utils.draw_bbox(frame, pred_bbox)

            twist,angular_z = get_move(W//2, x+(w//2) )
            
            # rostopic Rate에 따라 h 조정 (주석 처리해도 됩니다.)
            # 해당 부분에서 일시적인 토픽 전송 대기 발생
            #r.sleep()


        elif num_objects <1 :
            occlusion = True
            twist.linear.x =0
            twist.angular.z =0
            pub.publish(twist)
            time_stop = time.time()
            check=True
            # while time.time() - time_stop < 2 : 
            #     cv2.destroyAllWindows()
            #     cap.release()
            #     check = False
            #     break
            # if not check :
            #     break

        z_bias = 0.2 
        # tracker가 급격하게 움직인 경우 이전 angular를 사용한다.
        if abs(angular_z_pre) - abs(angular_z) > 0.35 + z_bias: 
            print(f"occlusion 발생  :  {occlusion }")
            occlusion = not occlusion
            time_wait_pub(1, pub,linear=0.3, angular=angular_z_pre)

        if not occlusion :
        # ================== new hand detection move check ===================
        # ================   hand를 while 내에서 계속 돌려볼 시 아래 사용 (2) ==========
            #if action !='stop' :
            print(" not occlusion, 정상 publish")
            pub.publish(twist)

        occlusion = False

        angular_z_pre = angular_z

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        
        #if len(bboxes) >0 :
        #result = utils.draw_bbox(frame, pred_bbox) /

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('result', result)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            cap.release()
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':

    rospy.init_node('scout_ros_detector_test', anonymous=False)
    pub = rospy.Publisher('/cmd_vel', 
        Twist, 
        queue_size=10)
    video_path = -1
    main(video_path,pub)
