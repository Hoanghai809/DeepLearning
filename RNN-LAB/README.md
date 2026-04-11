# Mạng Nơ-ron Hồi quy (RNN)
## Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **Python** | 3.11+ | Ngôn ngữ lập trình chính |
| **PyTorch** | 2.0+ | Framework xây dựng và huấn luyện mô hình RNN |
| **NumPy** | 1.24+ | Xử lý tính toán số học, tạo dữ liệu |
| **Pandas** | 2.0+ | Quản lý và thao tác dữ liệu dạng bảng |
| **Matplotlib** | 3.7+ | Trực quan hóa dữ liệu và kết quả |
| **scikit-learn** | 1.3+ | Chuẩn hóa dữ liệu (MinMaxScaler), đánh giá mô hình |
| **CUDA** | 12.0+ | Tăng tốc huấn luyện trên GPU |

## Cách hoạt động

### 1. Luồng dữ liệu trong mô hình:

1. **Input** → `(seq_length, 3 features)`
2. **RNN Layer** → `hidden_size = 32`
3. **Dropout** → `p = 0.2`
4. **Fully Connected Layer** → Linear transformation
5. **Output** → `1 giá trị dự đoán`

### 2. Quá trình huấn luyện
Mỗi epoch:

Forward pass: Tính output từ input sequence

Tính loss (MSE) giữa output và target

Backward pass: Tính gradient

Cập nhật trọng số (Adam optimizer)

Lưu train loss và validation loss


## Kết quả

### 1. Kết quả huấn luyện

| hidden_size | Final Train Loss | Final Val Loss |
|-------------|-----------------|----------------|
| 16 | 0.018365 | 0.005009 |
| 32 | 0.009936 | 0.002984 |
| 64 | 0.006803 | 0.003125 |

**Mô hình tốt nhất:** hidden_size = 32

### 2. Kết quả trên tập test (hidden_size=32)

| Chỉ số | Giá trị |
|--------|---------|
| **MSE** | 0.004503 |
| **MAE** | 0.053658 |
| **RMSE** | 0.067108 |

### 3. Biểu đồ Loss

<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/184629ef-8b2f-430b-8310-ab685cff75b7" />


- Train loss giảm từ ~0.09 xuống ~0.01
- Validation loss giảm từ ~0.04 xuống ~0.003
- Không có hiện tượng overfitting rõ rệt

### 4. Biểu đồ dự đoán vs Thực tế

<img width="861" height="554" alt="image" src="https://github.com/user-attachments/assets/430afebd-b736-4fb8-bead-1b3f59ba1dd5" />

- Dự đoán bám sát giá trị thực tế
- Sai số tập trung ở các điểm biến động mạnh

### 5. So sánh các tham số

| Tham số | Giá trị tốt nhất | Nhận xét |
|---------|-----------------|----------|
| seq_length | 10 | Ngắn quá → thiếu ngữ cảnh |
| seq_length | 20 | **Tối ưu** |
| seq_length | 30 | Dài quá → khó hội tụ |
| learning_rate | 0.001 | **Tối ưu** |
| learning_rate | 0.01 | Lớn quá → loss dao động |
| hidden_size | 32 | **Tối ưu** |

### 6. Dự đoán nhiều bước (Multi-step)

Dự đoán 5 bước tiếp theo từ cuối tập test:

| Bước | Giá trị dự đoán |
|------|----------------|
| 1 | -0.438628 |
| 2 | -0.195030 |
| 3 | 0.198141 |
| 4 | 0.513607 |
| 5 | 0.647099 |

### 7. Thống kê sai số

| Chỉ số | Giá trị |
|--------|---------|
| Sai số trung bình | 0.049883 |
| Sai số lớn nhất | 0.145859 |
| Sai số nhỏ nhất | 0.000166 |

## Nhận xét chung

✅ **Mô hình hoạt động tốt** với MSE = 0.0045 trên tập test

✅ **GPU RTX 5060 Ti** giúp huấn luyện nhanh

✅ **Hidden_size = 32** là lựa chọn tối ưu cho bài toán này

⚠️ **Hạn chế**: RNN cơ bản vẫn gặp vấn đề gradient vanishing với chuỗi dài

💡 **Cải thiện**: Có thể thay thế bằng LSTM hoặc GRU để học phụ thuộc dài hạn tốt hơn
