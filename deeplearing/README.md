
# **DeepLearning**
## Công nghệ sử dụng
***
### **Core Framework**
#### - PyTorch 2.9.1+ - Deep learning framework chính
#### - Python 3.11.9 - Ngôn ngữ lập trình
#### - Jupyter Notebook - Môi trường phát triển
***
### Data Processing & Visualization
#### - NumPy - Tính toán số học và mảng
#### - Pandas - Xử lý dữ liệu dạng bảng
#### - Matplotlib - Vẽ đồ thị và visualization
#### - Scikit-learn - Tiền xử lý dữ liệu
***
### Dataset
#### - Iris Dataset - 150 mẫu với 4 đặc trưng, 3 lớp hoa
#### - Synthetic Data - Dữ liệu giả lập cho regression
***
##  Cách Hoạt Động
### Xử Lý Dữ Liệu Pipeline
#### Raw Data → Preprocessing → Train/Test Split → PyTorch Tensors

### Tính Toán Gradient Tự Động
#### Sử dụng ```requires_grad=True``` để bật tính gradient
#### Tự động xây dựng computational graph
#### ```backward()``` tự động tính gradient qua chain rule

### Gradient Descent Optimization
```
for epoch in range(epochs):
    # Forward pass
    y_pred = W * x + b
    
    # Compute loss (MSE)
    loss = torch.mean((y_pred - y)**2)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()
```
### Memory Management
#### Chia sẻ bộ nhớ: torch.from_numpy() - cùng vùng nhớ với NumPy
#### Sao chép bộ nhớ: torch.tensor() - tạo bản sao độc lập

### Tensor Operations
#### Tạo tensor: empty(), zeros(), ones(), rand()
#### Reshape: view(), view_as(), reshape()
#### Mathematical operations: element-wise và matrix
***
## Kết Quả
### BTVN 1: Gradient Calculation
#### điểm cực trị tại ```x = ±0.3736```
#### Gradient nhỏ nhất: ```-3.0 (tại x=0)```
***
### BTVN 2: Gradient Descent
#### Giảm 84% giá trị hàm: Từ ```y=27.0``` xuống ```y=4.39```
#### Hội tụ sau 10 vòng với ```lr=0.01```
#### X giảm từ ```2.0``` → ```0.53```
***
### BTVN 3: Linear Regression
#### R² Score: ```0.9534``` 
#### Tham số học được:```W=3.30```, ```b=3.06``` (gần giá trị thực 3.0, 5.0)
#### Final Loss: ```2.803```
#### Dự đoán chính xác trên test points
***
### BTVN 4: Memory Management
#### Hiểu sự khác biệt ```torch.from_numpy()``` vs ```torch.tensor()```
#### Ứng dụng memory sharing/copying
***
### BTVN 5: Tensor Creation
#### tạo tensor
#### Thao tác reshape 
























