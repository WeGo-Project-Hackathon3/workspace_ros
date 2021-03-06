# ============================ depth cam =====================================
# depth cam으로 얻은 길이가 일정 범위안에 포함 시 True, False return 및 최소 길이 return 
def depth_action(box, depth_frame ) :
    # distance 1m내인지 여부 판단 
    w_coverage = [0]*box[2]
    h_coverage = []
    depth_dist =[]
    DISTANCE = 0.0
    try: 
        for ih in range(0,box[3], box[3]//10 ):
            for iw in range(0,box[2] , box[2]//10):
                dist = depth_frame.get_distance(box[0]+iw, box[1]+ih)
                if 1.45 < dist and dist < 4.8:
                    w_coverage[iw] = 1
                else :
                    depth_dist.append(dist)
            
            h_coverage.append(sum(w_coverage) > 0)                  
            w_coverage = [0]*box[2]
        DISTANCE = min(depth_dist) 
        if any(h_coverage)  : # 1.45 ~ 4.8 범위 이내 
            return True, DISTANCE
        else :
            return False, DISTANCE
    except:
        return False, 0