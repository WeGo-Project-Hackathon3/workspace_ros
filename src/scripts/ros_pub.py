import time
from geometry_msgs.msg import Twist

# ros 동작 publish 관련 함수
def get_move(W,track_w, hand_center, H=0, track_H=0, init_chk = False ):
    twist = Twist()
    z_bias_1 = 0.05


    # angular cma의 중앙을 0으로 둔 후 가장 왼쪽과 오른쪽을 -1, 1로 두는 계산을한다. 
    angular_z = round(-1.0 * (W-track_w)/W,1  )

    # 사람이 서있을때의 초기 보정값 
    if init_chk:
        init_hand_angular = round(1.0 * (W-hand_center)/W,1  )
        angular_z += init_hand_angular

    angular_chk = 1  #  0 : left  ,   1  : right
    if angular_z < 0 :  # 우회전 해야함 
        angular_chk = -1
    
    # 최소값 정의 angular
    if abs(angular_z) <= 0.18 :#3+ abs(z_bias_1) :
        angular_z = 0.0
    # 최대 값 정의 angular 
    elif abs(angular_z) > 0.5 :
        angular_z = 0.68 * angular_chk
    else : 
        angular_z +=z_bias_1
    
    # linear_x 고정값 
    # 실내에서 동작하는 경우 기준
    linear_x =  0.34

    #print("angular_z : ", angular_z) # cam 상에서 내 위치 : - right   + left   실제 적용해야하는 값 : +  right,    - left  (화면이 반대이므로)
    #print("linear_x : ", linear_x) # - right   + left서

    
    twist.linear.x = linear_x
    twist.angular.z = angular_z 

    return twist, angular_z, linear_x

# twist 값 초기화
def init_twist():
    twist = Twist()
    twist.linear.x = 0
    twist.linear.y = 0
    twist.linear.z = 0

    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = 0
    return twist

# 매개변수로 받은 linear, agnular 값을 발행 후 일정시간 대기 
def time_wait_pub(duration, pub,linear=0.2, angular=0.3):
    twist = Twist()
    twist.linear.x = linear-0.1; twist.linear.y = 0; twist.linear.z = 0

    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = angular
    time_chk = time.time()
    pub_chk = False
    while time.time() - time_chk > duration : 
        #rospy.sleep(0.2)
        if not pub_chk :
            pub.publish(twist)
            pub_chk = True 



# =============================== hand option에 따라 움직임을 재어할때 사용 ===============================
def hand_option(action,pub) :
    twist = Twist()

    if action == 'move':
        twist.linear.x = 0.3; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0

    elif action == "stop" :
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        
    pub.publish(twist)
