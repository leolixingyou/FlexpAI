# src/yolopv2_model.py

import torch
import cv2
import numpy as np
import rospy
from utils import letterbox, split_for_trace_model, non_max_suppression, lane_line_mask

class YOLOPv2:
    def __init__(self, weights='/workspace/src/perception/src/visual/yolopv2.pt', device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(weights)
        self.model.to(self.device)
        self.model.eval()
        rospy.loginfo("YOLOPv2 모델이 초기화되었습니다.")

    def detect(self, img):
        img_resized, _, _ = letterbox(img, new_shape=640, stride=32, auto=True)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.transpose(2, 0, 1)
        img_rgb = np.ascontiguousarray(img_rgb)
        img_tensor = torch.from_numpy(img_rgb).to(self.device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(img_tensor)
            pred = split_for_trace_model(pred, anchor_grid)
            pred = non_max_suppression(pred, 0.3, 0.45)
            
        return pred, seg, ll
