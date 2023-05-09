from models.models import trainer
from tests.config import *

if __name__ == '__main__':
    mytrainer = trainer(learning_rate= 0.001)
    mytrainer.load(name = 'best_loss_proba')
    # mytrainer.train(epoch= 100,early_stop_patience= 2)
    # mytrainer.save(name = 'test_loss')
    # 我们将训练和测试集分开
    mytrainer.get_show_data(index= 1)
    mytrainer.show(1)

# loss_test 0.0420  #当前大致的最好结果