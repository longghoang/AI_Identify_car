import os
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Đường dẫn đến thư mục chứa các nhãn thực tế
labels_path = '../test/labels'

# Đường dẫn đến thư mục chứa các ảnh kiểm thử
images_path = '../test/images'

# Tạo danh sách y_test
y_test = []
image_file_map = {}

for label_file in os.listdir(labels_path):
    if label_file.endswith('.txt'):
        image_name = label_file.replace('.txt', '.jpg')  # Giả định rằng tên file ảnh trùng với tên file nhãn
        with open(os.path.join(labels_path, label_file), 'r') as f:
            labels = f.readlines()
            for label in labels:
                class_id = int(label.split()[0])  # Lớp đối tượng là phần tử đầu tiên trong mỗi dòng
                y_test.append(class_id)
                if image_name not in image_file_map:
                    image_file_map[image_name] = []
                image_file_map[image_name].append(class_id)

# Tải mô hình YOLOv8 đã huấn luyện
model = YOLO('../data/best3.pt')

# Tạo danh sách y_scores
y_scores = []

for image_file in os.listdir(images_path):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        img_path = os.path.join(images_path, image_file)
        
        # Chạy mô hình dự đoán
        results = model.predict(img_path)
        
        # Xử lý kết quả dự đoán
        if image_file in image_file_map:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0]  # Lấy giá trị confidence score
                    y_scores.append(conf.item())
                # Append zeros if there are no detections to ensure lengths match
                while len(y_scores) < len(image_file_map[image_file]):
                    y_scores.append(0)  

# Đảm bảo y_scores và y_test có cùng chiều dài
if len(y_test) != len(y_scores):
    min_len = min(len(y_test), len(y_scores))
    y_test = y_test[:min_len]
    y_scores = y_scores[:min_len]

# Tính toán các giá trị FPR và TPR
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Vẽ đường cong ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
