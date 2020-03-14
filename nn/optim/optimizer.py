class Optimizer(object):
    '''
    Base class for all kinds of optimizers
    '''

    def __init__(self, model, lr):
        self.model = model
        self.module_num = len(self.model.modules)
        self.lr = lr

    def backward(self):
        NotImplementedError

    def update(self):
        NotImplementedError

