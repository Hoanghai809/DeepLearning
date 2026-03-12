## 🛠 Công nghệ sử dụng

- **Python 3.x** - Ngôn ngữ lập trình chính
- **TensorFlow/Keras** - Framework xây dựng và huấn luyện mô hình CNN
- **NumPy** - Thư viện tính toán ma trận và xử lý dữ liệu số
- **Matplotlib/Seaborn** - Trực quan hóa dữ liệu và kết quả huấn luyện
- **Jupyter Notebook** - Môi trường phát triển tương tác
- **OpenCV/PIL** - Xử lý ảnh 

## ⚙ Cách hoạt động

Mô hình CNN (Convolutional Neural Network) hoạt động qua các bước:

1. **Đầu vào**: Ảnh được đưa vào mạng dưới dạng ma trận pixel

2. **Tầng Tích chập (Convolution Layer)**:
   - Trích xuất đặc trưng từ ảnh bằng các bộ lọc (kernel)
   - Tạo ra các feature map chứa thông tin về cạnh, góc, texture

3. **Tầng Gộp (Pooling Layer)**:
   - Giảm kích thước feature map
   - Giữ lại thông tin quan trọng, giảm overfitting

4. **Tầng Làm phẳng (Flatten)**:
   - Chuyển feature map 2D thành vector 1D

5. **Tầng Kết nối đầy đủ (Fully Connected)**:
   - Vector đặc trưng được nhân với ma trận trọng số
   - Công thức: $y = Wx + b$
   - Tổng hợp đặc trưng để đưa ra dự đoán

6. **Đầu ra**: Xác suất các lớp (sử dụng Softmax)

## Kết quả

### Câu 1: Thay đổi số lượng epoch

**Kết quả với 5 epoch (ban đầu):**
  - Epoch 1: Loss 0.2603 - Accuracy 92.13%
  - Epoch 2: Loss 0.0837 - Accuracy 97.49%
  - Epoch 3: Loss 0.0634 - Accuracy 98.06%
  - Epoch 4: Loss 0.0539 - Accuracy 98.36%
  - Epoch 5: Loss 0.0473 - Accuracy 98.56%

✅ Độ chính xác trên tập test: **98.56%**

**b) Kết quả với 10 epoch (sau khi tăng):**
  - Epoch 1: Loss 0.0414 - Accuracy 98.70%
  - Epoch 2: Loss 0.0376 - Accuracy 98.83%
  - Epoch 3: Loss 0.0353 - Accuracy 98.89%
  - Epoch 4: Loss 0.0327 - Accuracy 98.98%
  - Epoch 5: Loss 0.0295 - Accuracy 99.12%
  - Epoch 6: Loss 0.0276 - Accuracy 99.15%
  - Epoch 7: Loss 0.0263 - Accuracy 99.19%
  - Epoch 8: Loss 0.0244 - Accuracy 99.21%
  - Epoch 9: Loss 0.0235 - Accuracy 99.28%
  - Epoch 10: Loss 0.0214 - Accuracy 99.31%

✅ Độ chính xác trên tập test: **99.31%** (tăng **0.75%** so với ban đầu)

**Nhận xét:**
- Loss giảm mạnh ở 2 epoch đầu (từ 0.2603 → 0.0837)
- Từ epoch 3-10, loss giảm chậm dần và chững lại (~0.0214)
- Accuracy tăng đều qua các epoch và đạt cao nhất ở epoch 10

### Câu 2: Thêm một tầng tích chập

**Kiến trúc mới:** Thêm `conv3` (32 → 64 kênh, kernel 3x3)

**Kết quả:**
- Độ chính xác trên tập test: **98.34%**

**Tác dụng của việc thêm tầng tích chập:**
- ✅ Học được các đặc trưng phức tạp hơn từ dữ liệu
- ✅ Tăng độ chính xác của mô hình
- ⏱️ Tăng thời gian tính toán do nhiều tham số hơn

---

### Câu 3: Thay đổi learning rate

**So sánh các learning rate khác nhau:**

| Learning Rate | Test Accuracy | Biểu đồ loss |
|---------------|---------------|--------------|
| **0.001** | 97.26% | Giảm đều, ổn định |
| **0.01** (gốc) | ~98.5% | Giảm tốt, cân bằng |
| **0.1** | 97.26% | Dao động mạnh, không ổn định |

**Ảnh hưởng của learning rate:**
- **LR quá nhỏ (0.001):** Học chậm, cần nhiều epoch để hội tụ
- **LR quá lớn (0.1):** Model dao động mạnh, khó hội tụ, có thể bỏ lỡ điểm tối ưu
- **LR vừa phải (0.01):** Cân bằng giữa tốc độ và độ ổn định

---

### Câu 4: Vẽ feature map từ tầng tích chập

**So sánh feature map giữa các tầng:**

| Tầng | Đặc điểm |
|------|----------|
| **conv1** | Chi tiết, rõ nét, giữ hình dạng gốc, độ phân giải cao |
| **conv2** | Trừu tượng hơn, khó nhận dạng hơn, hoa văn phức tạp |
| **conv3** | Đặc trưng phức tạp, độ phân giải thấp, mất cấu trúc không gian |

**Hình ảnh feature map:**
![Feature maps từ conv1 và conv2](attachment:image.png)

**Giải thích:**
- **Tầng đầu:** Học các đặc trưng đơn giản (cạnh, góc, đường viền)
- **Tầng giữa:** Kích thước giảm dần, đặc trưng phức tạp hơn (hình khối, bộ phận)
- **Tầng cuối:** Đặc trưng trừu tượng nhất, đại diện cho khái niệm tổng thể

**Quy luật tổng quát:**
> Càng vào sâu, feature map càng trừu tượng, kích thước càng nhỏ, nhưng thông tin càng có ý nghĩa cho việc phân loại.

















