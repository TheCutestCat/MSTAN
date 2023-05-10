from Models.Seq2seq import trainer_seq2seq
from config import *
import time
if __name__ == '__main__':
    # ['wind10', 'temp', 'atmosphere', 'humidity', 'power', 'en_month','en_hour', 'en_cos_angle']
    # 我们只修改了很少的结果，看看效果如何。。
    mytrainer = trainer_seq2seq(learning_rate= 0.0004)
    mytrainer.load(name = 'seq2seq_MoreData_3_22')
    # mytrainer.train(epoch= 10,early_stop_patience= 2)
    # mytrainer.save(name = 'seq2seq_MoreData_test')
    mytrainer.show(index_begin= 100,index_end= 600)
    #终于match上了，但是感觉效果还是不是特别好（对数据中心laod一下啦）
    #无状态模型
    # the best loss is just about 5.4558 still a little high.
    # we can use the remote computer, this is fantastic