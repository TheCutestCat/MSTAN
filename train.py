from Models.Seq2seq import trainer_seq2seq
from config import *

if __name__ == '__main__':
    mytrainer = trainer_seq2seq(learning_rate= 0.01)
    mytrainer.load(name = 'seq2seq_test')
    mytrainer.train(epoch= 30,early_stop_patience= 2)
    mytrainer.save(name = 'seq2seq_test')
    # 我们将训练和测试集分开
    # mytrainer.get_show_data(index= 1)
    # mytrainer.show(1)