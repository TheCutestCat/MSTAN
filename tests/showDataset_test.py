from matplotlib import pyplot as plt
from data.TestDataLoader import loader_show,show
# this finally works
batch_size = 1
data_path = '../data/test_small_data.csv'
T0 = 32
tau = 32
index_begin = 100
index_end = 1000
dataloader = loader_show(data_path,T0,tau,index_begin,index_end)
y_save = []
iters = 0
for i,(en_x,en_x_pre,y) in enumerate(dataloader):
    y_ = y.squeeze().tolist()
    y_save += y_
    iters = i
print(f'with iter of {iters}')

y_save_compare = show(data_path,T0,index_begin,index_end)
plt.figure(figsize=(13, 6))
plt.plot(y_save, linestyle='-', label='real')
plt.plot(y_save_compare, linestyle = '-',label = 'compare')
# plt.ylim(0, 1)
plt.legend()
# plt.savefig('new.png')
plt.show()