import numpy as np


# ================== box간의 iou를 구한다 ===============================
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

# ================= iou를 기준으로 가장 큰 bounding box를 가진 person bbox return =======================
def get_coordinates(box, x, y, x1, y1):
	""" Get co-ordinates of flaged person
	"""
	if len(box) == 0:
		return
	iou_scores = []
	for i in range(len(box)):
		iou_scores.append(get_iou(box[i],[x,y,x1,y1]))

	index = np.argmax(iou_scores)
	print("get_coordinates : ", iou_scores, ' ',box, ' ', x, y, x1, y1)

	if np.sum(iou_scores) == 0:
		# print('#'*20, 'No Match found', '#'*20)
		box = np.array(box)
		distance = np.power(((x+x1)/2 - np.array(box[:,0] + box[:,2])/2),2) + np.power(((y+y1)/2 - (box[:,1]+box[:,3])/2), 2)
		index = np.argmin(distance)

	x, y, w, h = box[index][0], box[index][1], (box[index][2]-box[index][0]), (box[index][3]-box[index][1])
	initBB = (x+w//2-35,y+h//2-35,70,70)  # default  iou box
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
        return round(dist/W ,3)
    else :
        # 두 점사이의 거리
        dist = round( np.linalg.norm(boxA_c-boxB_c),2 )
        return round(dist/W ,3)


# ==================== bbox 간 일정 범위 이하 bbox만 남기기 =============================
# track_bbox, 전체 bbox를 받은 후 가장 거리가 덜 튕기는 box를 구한다.
# hand_bbox를 기준 점으로 보기 
def get_box_distance(bbox, track_box,  W, H=0):
    if len(bbox) == 0: # no person
        return 
    elif len(bbox) == 1 :
        return bbox
    select_box = []
    select_box_ratio = []
    bias_ratio = 0.18
    x,y,x1,y1 = track_box[0], track_box[1], track_box[0]+track_box[2], track_box[1]+track_box[3]  
    for i in range(len(bbox)):
        ratio =  get_box_dist(bbox[i], [x,y,x1,y1], W, H=0,isWidth=True )
        select_box_ratio.append(ratio)
        if ratio < bias_ratio :
            select_box.append(bbox[i])
    

    if len(select_box) == 0 :
        index = np.argmin( select_box_ratio )
        return np.array([bbox[index]])
    else :
        return np.array(list(select_box) )