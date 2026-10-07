# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

|             |                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------- |
| Họ và tên   | Nguyễn Mai Hoàng Thiện                                                                              |
| MSSV        | 2A202602912                                                                                         |
| Lớp / Khóa  | K4                                                                                                  |
| Repo GitHub | <https://github.com/nmht/K4-L3L4-Track2-Day21-NguyenMaiHoangThien-2A202602912-CI-CD-for-AI-Systems> |
| Ngày nộp    | 7/10/2026                                                                                           |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n\_estimators | learning\_rate | max\_depth | f1\_score | accuracy |
| -------- | ------------- | -------------- | ---------- | --------- | -------- |
| 1        | 100           | 0.10           | 3          | 0.7109    | 0.8780   |
| 2        | 50            | 0.05           | 2          | 0.6051    | 0.8460   |
| 3        | 200           | 0.10           | 5          | 0.7149    | 0.8740   |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số lần chạy 3 tốt hơn các bộ còn lại vì đạt `f1_score` trên lớp dương cao nhất (0.7149). Lần chạy có `accuracy` cao nhất là lần 1 (0.8780), nhưng lại có `f1_score` thấp hơn lần 3. Điều này cho thấy mô hình ở lần 1 tuy đoán đúng tổng thể nhiều hơn nhưng lại bỏ sót hoặc đoán sai nhiều hơn ở lớp thiểu số (>50K). Khi tăng số cây `n_estimators` và độ sâu `max_depth`, mô hình có khả năng học các ranh giới quyết định phức tạp hơn, làm tăng khả năng nhận diện lớp thiểu số, đánh đổi lại một chút độ chính xác chung.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Dữ liệu thu nhập phân bố mất cân bằng, lớp thu nhập > 50K thường chiếm tỷ lệ nhỏ (khoảng 24%). Nếu một mô hình tồi luôn dự đoán tất cả là "thu nhập thấp" (<=50K), thì `accuracy` vẫn có thể đạt khoảng 76%. Con số này gây hiểu nhầm về khả năng thực sự của mô hình. F1-score của lớp dương (thu nhập > 50K) đo lường sự cân bằng giữa độ chính xác (Precision) và độ phủ (Recall) trên chính đối tượng thiểu số mà ta quan tâm. Việc dùng `average="binary"` hay đánh giá trực tiếp trên lớp dương giúp đo lường sát nhất bài toán thực tế, trong khi dùng `average="weighted"` hay `average="macro"` sẽ pha trộn điểm số của lớp đa số, làm mờ đi hiệu năng thực sự trên lớp mục tiêu.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn                                            | Nguyên nhân                                                                     | Cách giải quyết                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Không thể mở `mlflow ui`                            | Phiên bản `sqlalchemy` (2.1.3+) xung đột với `mlflow==2.13.0`                   | Hạ cấp thư viện bằng cách cài đặt lại `sqlalchemy<2.0.30`.                                       |
| Lệnh train chạy nhưng không ghi log vào `mlflow.db` | Biến môi trường Tracking URI không được set đúng khi chạy lệnh python trực tiếp | Thêm trực tiếp `os.environ` vào đầu script `train.py` để trỏ cố định đến thư mục lưu trữ MLflow. |
| Chưa thể train qua DVC do cấu hình remote           | Google Drive yêu cầu xác thực bằng trình duyệt khi cấu hình remote DVC          | (Đang chuẩn bị thực hiện theo Bước 2 bằng Service Account GCP thay thế).                         |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

|                              | f1\_score | accuracy |
| ---------------------------- | --------- | -------- |
| Bước 2 (chỉ `train_batch1`)  | 0.7149 | 0.874|
| Bước 3 (thêm `train_batch2`) | 0.7354    | 0.882|

**Nhận xét:** Việc bổ sung thêm dữ liệu mới ở Bước 3 không tạo ra sự gia tăng đột phá đáng kể, đôi khi f1_score còn có thể dao động hoặc giảm nhẹ. Nguyên nhân là do hai tập dữ liệu (`train_batch1` và `train_batch2`) đều được trích xuất ngẫu nhiên từ cùng một nguồn nên có cùng chung một phân phối dữ liệu. Lượng dữ liệu ở Bước 2 (22.361 mẫu) vốn đã đủ lớn để mô hình học được hầu hết các quy luật cần thiết, do đó việc tăng gấp đôi lượng dữ liệu không mang lại thêm nhiều thông tin mới mang tính quyết định. Điều quan trọng nhất mà Bước 3 thể hiện là quy trình CI/CD hoàn toàn tự động đã hoạt động trơn tru.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- Bonus 1 - Tracking MLflow từ xa với DagsHub: \_\_\_
- Bonus 2 - Điều chỉnh ngưỡng quyết định: \_\_\_
- Bonus 3 - Báo cáo precision / recall tự động: \_\_\_
- Bonus 4 - Hoàn trả về phiên bản trước: \_\_\_
- Bonus 5 - Cảnh báo lệch lạc dữ liệu: \_\_\_
