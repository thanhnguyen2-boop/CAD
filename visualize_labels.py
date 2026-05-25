import cv2
import numpy as np
from pathlib import Path

# Cấu hình các lớp (classes) dựa trên data.yaml của bạn
CLASSES = ['front_view', 'side_view', 'section_view', 'isometric_view', 'note', 'table']

# Cấu hình màu cho từng lớp (B, G, R)
COLORS = [
    (255, 0, 0),    # front_view: Xanh dương
    (0, 165, 255),  # side_view: Cam
    (0, 255, 255),  # section_view: Vàng
    (0, 0, 255),    # isometric_view: Đỏ
    (255, 0, 255),  # note: Tím
    (0, 255, 0)     # table: Xanh lá
]

def visualize_yolo_labels(image_path, label_path, output_path):
    # Sử dụng np.fromfile để đọc ảnh an toàn với đường dẫn dài/Unicode
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return

    h, w = img.shape[:2]

    # Đọc file nhãn YOLO
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Không tìm thấy file nhãn: {label_path}")
        return

    # Vẽ từng khung
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        cls_id = int(parts[0])
        x_c, y_c, w_c, h_c = map(float, parts[1:5])
        
        # Chuyển đổi tọa độ YOLO (center_x, center_y, width, height) dạng chuẩn hóa (0.0-1.0)
        # sang tọa độ pixel tuyệt đối (x_min, y_min, x_max, y_max)
        x_min = int((x_c - w_c / 2) * w)
        y_min = int((y_c - h_c / 2) * h)
        x_max = int((x_c + w_c / 2) * w)
        y_max = int((y_c + h_c / 2) * h)
        
        # Lấy tên và màu của lớp
        class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class_{cls_id}"
        color = COLORS[cls_id % len(COLORS)]
        
        # Vẽ khung hình chữ nhật
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Vẽ nền cho text để dễ đọc hơn
        label_text = f"{cls_id}: {class_name}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x_min, y_min - text_h - 5), (x_min + text_w, y_min), color, -1)
        
        # In tên nhãn lên
        cv2.putText(img, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Lưu ảnh kết quả
    cv2.imwrite(output_path, img)
    print(f"Saved visualized image to: {output_path}")

if __name__ == "__main__":
    # Thay đổi đường dẫn này trỏ tới file cụ thể bạn muốn kiểm tra
    base_dir = Path(r"C:\Users\thanh.nguyen2\Documents\project-python\cad\technical_drawings_v6\train")
    
    # Tên file hiện tại bạn đang mở
    filename = "0ce07ea5662c521deafb4c1200f4611f_jpg.rf.1cc8490d1578210989485cd553731da3"
    
    img_file = base_dir / "images" / f"{filename}.jpg"
    lbl_file = base_dir / "labels" / f"{filename}.txt"
    out_file = Path(r"C:\Users\thanh.nguyen2\Documents\project-python\cad") / f"{filename}_visualized.jpg"
    
    visualize_yolo_labels(str(img_file), str(lbl_file), str(out_file))
