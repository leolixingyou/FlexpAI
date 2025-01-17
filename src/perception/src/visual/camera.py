# src/camera.py

import time
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


from yolopv2 import YOLOPv2


from utils import lane_line_mask, thin_lane_line_mask

class Camera:
    def __init__(self):
        rospy.init_node('hahah')

        # 쓰레드 안전을 위한 락 초기화
        self.lock = False
        self.latest_frame = None

        # 모델 초기화
        self.weights = '/workspace/src/perception/src/visual/yolopv2.pt'
        self.model = YOLOPv2(self.weights)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30) # 10hz

        # ROS 관련 설정
        rospy.Subscriber("/Ctrl_CV/camera/image", Image, self.callback)
        self.img2fusion = rospy.Publisher('/output_img', Image, queue_size=1)


    def callback(self, msg):
        if not self.lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono")
            self.lock = True

    def process_images(self):
        while not rospy.is_shutdown():
            frame = None
            if self.lock:
                frame = self.latest_frame
                start = time.time()
                self.process_image(frame)
                print(f'FPS is {1/(time.time()-start)} FPS')
                self.lock = False
            self.rate.sleep()

    def process_image(self, img0):
        # YOLOPv2 모델을 사용한 차선 감지 및 세그멘테이션
        pred, seg, ll = self.model.detect(img0)
        ll_seg_mask = lane_line_mask(ll)

        # 얇은 차선 마스크 적용
        ll_seg_mask = thin_lane_line_mask(ll_seg_mask)
        ll_seg_mask = cv2.cvtColor(ll_seg_mask, cv2.COLOR_GRAY2BGR)

        # # 8. 원본 영상에 투영된 결과 영상 출력
        # cv2.imshow("Inverse BEV with Lane Overlay", ll_seg_mask)

        # cv2.waitKey(1)

        output_msg = self.bridge.cv2_to_imgmsg(ll_seg_mask, encoding="bgr8")
        self.img2fusion.publish(output_msg)

if __name__ == "__main__":
    camera = Camera()
    camera.process_images()