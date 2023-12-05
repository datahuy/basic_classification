## Train classification model
1. Dataset định dạng `.csv` gồm 2 col: product name - tên sản phẩm (`sentences`) và label - nhãn (`labels`)

2. **Build the dataset** 
Chạy phần sau để extract data
```
python build_data.py --data_dir DATA_DIR --data_name FILE.CSV

```

Chạy xong dữ liệu sẽ tách làm 3 tập train/val/test set cho việc train

3. **Build vocabularies** 
```
python build_vocab.py --data_dir data/data_demo --min_count_word 1
```

Tạo bộ từ điển toàn bộ các từ`words.txt` và tạo bộ nhãn các danh mục phân loại `labels.txt`.

4. File config tuning model:
 `data/config.yaml`

5. **Train** 
```
python train.py --data_dir DATA_DIR --config_path data/config.yaml --category CATEGORY
```
6. **Understand model**
6.1 Encoder Class:
- Encoder là một lớp kế thừa từ nn.Module, đại diện cho bộ mã hóa của mô hình.
Trong hàm khởi tạo (__init__), các thông số như kích thước nhãn (label_size), kích thước ẩn (hidden_size), số từ tối đa (max_words), và xác suất dropout (dropout) được thiết lập.
- Mô hình có hai lớp tuyến tính (layer1 và layer2) với các tham số được khởi tạo và đặt lại trong hàm reset_parameters.
- Hàm forward định nghĩa quá trình truyển tiếp của mô hình, trong đó các phép toán tuyến tính và hàm kích hoạt ReLU được áp dụng.

6.2 ClassfierModel Class:
- ClassfierModel là lớp chính của mô hình phân loại.
Trong hàm khởi tạo (__init__), một đối tượng Encoder được tạo ra với các thông số từ tham số đầu vào và được lưu trong biến self.encoder.
- Hàm preprocess thực hiện xử lý trước dữ liệu đầu vào, chuyển đổi văn bản thành biểu diễn số và tạo tensor text_embeddings.
- Hàm predict dự đoán nhãn của dữ liệu đầu vào. Nó sử dụng mô hình encoder đã được huấn luyện và trả về danh sách dự đoán và xác suất tương ứng.
- Hàm encode trả về biểu diễn số của dữ liệu đầu vào, cũng sử dụng mô hình encoder.
- Hàm preprocess:
Hàm này chuyển đổi mỗi đầu vào văn bản thành biểu diễn số bằng cách sử dụng từ điển vocab_dict và các chỉ số của các từ trong từ điển. Các từ không xuất hiện trong từ điển được thay thế bằng chỉ số của từ UNK.
- Hàm predict:
Hàm này đưa ra dự đoán và xác suất tương ứng cho dữ liệu đầu vào, sử dụng mô hình encoder đã được huấn luyện.
- Hàm encode:
Hàm này trả về biểu diễn số của dữ liệu đầu vào, sử dụng mô hình encoder.