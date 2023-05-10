from Models.Seq2seq import trainer_seq2seq
from config import *
import time
if __name__ == '__main__':

    mytrainer = trainer_seq2seq(learning_rate= 0.0005)
    mytrainer.load(name = 'seq2seq_value_best_5_27')
    # mytrainer.train(epoch= 10,early_stop_patience= 4)
    # mytrainer.save(name = 'seq2seq_test')
    mytrainer.show(index_begin= 350,index_end= 600) #终于match上了，但是感觉效果还是不是特别好（对数据中心laod一下啦）
    #record
    # the best loss is just about 5.4558 still a little high.
    # we can use the remote computer, this is fantastic