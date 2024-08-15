import os
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize

# Đường dẫn đến thư mục chứa các nhãn thực tế
labels_path = './test/labels'

# Đường dẫn đến thư mục chứa các ảnh kiểm thử
images_path = './test/images'

# Tạo danh sách y_test và y_scores
y_test = []
y_scores = []
image_file_map = {}

# Tạo một set để lưu tất cả các lớp đã có trong dữ liệu
class_set = set()

for label_file in os.listdir(labels_path):
    if label_file.endswith('.txt'):
        image_name = label_file.replace('.txt', '.jpg')  # Giả định rằng tên file ảnh trùng với tên file nhãn
        with open(os.path.join(labels_path, label_file), 'r') as f:
            labels = f.readlines()
            for label in labels:
                class_id = int(label.split()[0])  # Lớp đối tượng là phần tử đầu tiên trong mỗi dòng
                class_set.add(class_id)
                y_test.append(class_id)
                if image_name not in image_file_map:
                    image_file_map[image_name] = []
                image_file_map[image_name].append(class_id)

# Tải mô hình YOLO đã huấn luyện
model = YOLO('../data/best3.pt')

for image_file in os.listdir(images_path):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        img_path = os.path.join(images_path, image_file)
        
        # Chạy mô hình dự đoán
        results = model.predict(img_path)
        
        # Xử lý kết quả dự đoán
        if image_file in image_file_map:
            temp_scores = [0] * len(class_set)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0]  # Lấy giá trị confidence score
                    class_id = int(box.cls[0])
                    temp_scores[class_id] = max(temp_scores[class_id], conf.item())
            y_scores.append(temp_scores)

# Chuyển đổi y_test thành dạng one-hot để sử dụng trong multiclass ROC
y_test_binarized = label_binarize(y_test, classes=list(class_set))

# Đảm bảo y_scores và y_test có cùng chiều dài
if len(y_test_binarized) != len(y_scores):
    min_len = min(len(y_test_binarized), len(y_scores))
    y_test_binarized = y_test_binarized[:min_len]
    y_scores = y_scores[:min_len]

# Tính toán các giá trị FPR, TPR cho từng lớp và vẽ ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_set)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], [score[i] for score in y_scores])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Vẽ đường cong ROC cho từng lớp
plt.figure()
for i in range(len(class_set)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Each Class')
plt.legend(loc="lower right")
plt.show()
