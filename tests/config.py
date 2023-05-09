
data_path = 'data/test_small_data.csv'
index_wind = ['wind10', 'wind30', 'wind50']
index_other = ['angle10', 'angle30', 'angle50', 'temp', 'atmosphere', 'humidity']
T0 = 48
tau = 48
batch_size = 32
M_wind = 3
M_other = 6

sequence_length_in = T0
sequence_length_out = tau
input_size = 2
output_size = 2  # 只去输出一个对应的风速大小
hidden_size = 64

input_dim = hidden_size
d_attention = 16
output_dim = d_attention
seq_len = T0 + tau
m = 1

# learning_rate = 0.001 #直接在模型那边去进行定义