import os
import cv2
import yaml
import numpy as np

class BBoxDrawer:
    def __init__(self, yaml_path):
        self.classes = self.load_yaml_classes(yaml_path)
        self.colors = self.generate_colors(len(self.classes))

    def load_yaml_classes(self, yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data['names']

    def generate_colors(self, num_classes):
        # 生成均匀分布的HSV颜色
        colors = []
        for i in range(num_classes):
            # H值在0-180之间均匀分布，S和V固定为255
            hue = int(180 * i / num_classes)
            # 转换HSV到BGR
            rgb = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, rgb)))
        return colors

    def draw_bbox(self, image, bbox, class_id):
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
        x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
        
        color = self.colors[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{self.classes[class_id]}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def process_image(self, image_path, label_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return None
        
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    self.draw_bbox(image, (x, y, w, h), int(class_id))
                except ValueError as e:
                    print(f"Error processing label in {label_path}: {e}")
                    continue
        return image

def main():
    base_dir = '/workspace/ultralytics/datasets/s2tld/'
    yaml_path = os.path.join(base_dir, 'bs_test.yaml')  # YAML文件路径
    for name in ['train', 'val']:
        images_dir = os.path.join(base_dir, f'{name}/images')
        labels_dir = os.path.join(base_dir, f'{name}/labels')
        output_dir = os.path.join(base_dir, f'{name}/output')
        
        os.makedirs(output_dir, exist_ok=True)
        
        drawer = BBoxDrawer(yaml_path)
        
        for image_file in os.listdir(images_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_dir, image_file)
                label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')
                
                if os.path.exists(label_path):
                    processed_image = drawer.process_image(image_path, label_path)
                    if processed_image is not None:
                        output_path = os.path.join(output_dir, image_file)
                        cv2.imwrite(output_path, processed_image)
                        print(f"Processed: {image_file}")
                else:
                    print(f"No label file found for: {image_file}")

if __name__ == "__main__":
    main()