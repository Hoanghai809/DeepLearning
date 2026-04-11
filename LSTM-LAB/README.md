#  Long Short-Term Memory (LSTM) 

## 1. Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **Python** | 3.10+ | Ngôn ngữ lập trình chính |
| **PyTorch** | 2.0+ | Thư viện deep learning, xây dựng mô hình LSTM |
| **NumPy** | 1.24+ | Xử lý mảng số, tạo dữ liệu |
| **Matplotlib** | 3.7+ | Vẽ biểu đồ, trực quan hóa kết quả |
| **scikit-learn** | 1.3+ | Chuẩn hóa dữ liệu (MinMaxScaler) |

## 2. Cách hoạt động

### 2.1. Quy trình xử lý
Bước 1: Nhận đầu vào x_t và hidden state cũ h_{t-1}
Bước 2: Tính toán các cổng (forget, input, output)
Bước 3: Cập nhật Cell State: C_t = f_t × C_{t-1} + i_t × tanh(...)
Bước 4: Tính hidden state mới: h_t = o_t × tanh(C_t)
Bước 5: Đưa ra đầu ra và truyền sang bước tiếp theo

### 2.2. Ứng dụng

| Bài toán | Đầu vào | Đầu ra | Ứng dụng thực tế |
|----------|---------|--------|------------------|
| **Chuỗi thời gian** | 10 giá trị quá khứ | 1 giá trị tương lai | Dự báo giá cổ phiếu, nhiệt độ |
| **Dự đoán từ** | 2-3 từ trước | 1 từ tiếp theo | Gợi ý từ, hoàn thành câu |

## 3. Kết quả

### 3.1. Bài toán dự đoán chuỗi thời gian

**Dữ liệu**: Chuỗi sin + nhiễu (300 điểm)

| Thông số | Giá trị |
|----------|---------|
| seq_length (số bước quá khứ) | 10 |
| Tập huấn luyện | 80% |
| Tập kiểm tra | 20% |
| Số epoch | 200 |

**Kết quả huấn luyện:**

| Epoch | Loss |
|-------|------|
| 0 | 0.5219 |
| 50 | 0.0158 |
| 100 | 0.0136 |
| 150 | 0.0125 |

**Đánh giá:**
- ✅ Mô hình học được xu hướng và tính chu kỳ của dữ liệu
- ✅ Loss giảm dần và ổn định
- ✅ Đường dự đoán bám sát đường thực tế

### 3.2. Bài toán dự đoán từ tiếp theo

**Dữ liệu**: Câu tiếng Việt đơn giản

| Thông số | Giá trị |
|----------|---------|
| Kích thước từ điển | 7 từ |
| embedding_dim | 8 |
| hidden_dim | 16 |
| Số epoch | 100 |

**Kết quả dự đoán:**

| Đầu vào | Dự đoán | Kết quả |
|---------|---------|---------|
| "tôi thích nghe" | "nhạc" | ✅ Đúng |
| "tôi thích xem" | "phim" | ✅ Đúng |
| "thích nghe" | "nhạc" | ✅ Đúng |

**Đánh giá:**
- ✅ Mô hình học được mối quan hệ "thích nghe → nhạc"
- ✅ Mô hình học được mối quan hệ "thích xem → phim"
- ✅ Dự đoán chính xác 100% trên tập huấn luyện

### 3.3. Nhận xét chung

**Ưu điểm:**
- LSTM giải quyết được vấn đề vanishing gradient
- Học được các phụ thuộc dài hạn trong dữ liệu
- Dễ cài đặt với PyTorch

**Hạn chế:**
- Cần nhiều dữ liệu để học tốt
- Thời gian huấn luyện lâu hơn RNN thông thường
- Khó điều chỉnh siêu tham số

**Hướng cải thiện:**
- Tăng số lượng dữ liệu
- Điều chỉnh hidden_size, num_layers
- Thêm dropout để tránh overfitting
- Sử dụng bidirectional LSTM

## 4. Cài đặt nhanh

```bash
pip install torch numpy matplotlib scikit-learn
