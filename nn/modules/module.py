class Module(object):
    def __init__(self):
        self.modules = list()
        NotImplementedError

    def forward(self, *args, **kwargs):
        NotImplementedError

    def backward(self, *args, **kwargs):
        NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
