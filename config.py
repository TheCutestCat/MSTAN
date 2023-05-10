
data_path = 'data/clean_data_20000.csv' #注意去掉末尾的 ‘.csv’
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

input_dim = 32 #hidden_size/2 因为添加上了glu模块
d_attention = 16
output_dim = 16 #d_attention/2
input_mix = 8 #output_dim/2
seq_len = T0 + tau
m = 1

# learning_rate = 0.001 #直接在模型那边去进行定义