from models.models import trainer
from tests.config import *

if __name__ == '__main__':
    mytrainer = trainer(learning_rate= 0.000001)
    mytrainer.load(name = 'best_loss_0.061')
    mytrainer.train(epoch= 15,early_stop_patience= 2)
    mytrainer.save(name = 'test_loss')
    # 我们将训练和测试集分开
    # mytrainer.get_show_data()
    # mytrainer.show(1)

# loss_train 1.06 loss_test 0.122  #一个大概的参数