from nn.modules import Linear, Conv, Pooling
from .optimizer import Optimizer

class MBGD(Optimizer):
    def __int__(self, *args, **kwargs):
        super(MBGD, self).__int__(*args, **kwargs)

    def backward(self, loss):
        for idx in reversed(range(self.module_num)):
            module = self.model.modules[idx]
            if idx == self.module_num - 1:
                module.backward(loss)
            else:
                if isinstance(self.model.modules[idx+1], Linear):
                    module.backward(self.model.modules[idx+1].delta, self.model.modules[idx+1].weight)
                elif isinstance(self.model.modules[idx+1], Conv):
                    module.backward(self.model.modules[idx+1].delta, self.model.modules[idx+1].weight,
                                    self.model.modules[idx+1].padding)
                elif isinstance(self.model.modules[idx+1], Pooling):
                    module.backward(self.model.modules[idx+1].delta, pos_top=self.model.modules[idx+1].pos)

    def update(self, x):
        for idx in range(self.module_num):
            module = self.model.modules[idx]
            if idx == 0:
                module.update(x, self.lr)
            else:
                module.update(self.model.modules[idx-1].output, self.lr)
