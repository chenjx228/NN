import nn


class BPNN(nn.Module):
    def  __init__(self, dim_in, dim_out):
        super(BPNN, self).__init__()

        self.linear1 = nn.Linear(dim_in, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, dim_out)

        self.modules = [self.linear1, self.linear2, self.linear3]

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, 1)
        x = self.linear1(x)
        # print(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x
