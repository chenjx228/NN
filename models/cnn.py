import nn


class CNN(nn.Module):
    def __init__(self, dim_in, dim_out, height, width):
        super(CNN, self).__init__()

        hidden_dim = 1
        self.conv1 = nn.Conv(dim_in=dim_in, dim_out=hidden_dim, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool(kernel_size=3, padding=1, stride=2)
        self.linear1 = nn.Linear(int(hidden_dim*height*width/4), 20)
        self.linear2 = nn.Linear(20, dim_out)

        self.modules = [self.conv1, self.maxpool, self.linear1, self.linear2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1, 1)
        x = self.linear1(x)
        x = self.linear2(x)

        return x
