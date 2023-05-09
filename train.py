from Models.MSTAN import trainer_MSTAN_proba
from config import *

if __name__ == '__main__':
    mytrainer = trainer_MSTAN_proba(learning_rate= 0.0005)
    mytrainer.load(name = 'test_loss')
    mytrainer.train(epoch= 10,early_stop_patience= 2)
    mytrainer.save(name = 'test_loss')
    # 我们将训练和测试集分开
    # mytrainer.get_show_data(index= 1)
    # mytrainer.show(1)