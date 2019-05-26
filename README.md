# Task11_Intern
LSTM 

1. Initial data is in raw_dataset file
  - test_para.csv
  - test_ques.csv
  - train_label.csv
  - train_para.csv
  - train_ques.csv

2. Assignment
- problem_description.jpg

3. Code
- load_data.py
- model.py
- train.py
- test.py
- step0_data_preprocessing.ipynb



x [32, 10]  h [32, 3]

x.unsqueeze(1)  [32, 1, 10]

h.Linear(3, 256)  [32, 256]

h.unsqueeze(0) [1, 32, 256]

h.expand().contiguous() [3, 32, 256]  

x.conv1(1, 128)  [32, 128, 8]

x.permute(0, 2, 1) [32, 8, 128]   #相当于一句话有8个单词，每个单词表示成128维向量

lstm(x, (h, h))   # input_size, hidden_dim, num_layers
				  # 128,        256,        3
				  # h
				  # num_layers, batch, output_size
				  # 3,          32,    256

r_out, (h_n, h_c)

r_out [32, 8, 256]

r_out[:, -1, :]   [32, 256]

r_out[:,-1,:].Linear(256, 1)  [32, 1]

...sigmoid() [32, 1]
