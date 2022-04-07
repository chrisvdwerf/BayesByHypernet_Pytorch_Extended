from util.models.bbb import *
import torch as th


class BBBNet(th.nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_weight_samples):
        super().__init__()
        self.bbb1 = BBBLayer(n_inputs, n_hidden, n_weight_samples)
        self.bbb2 = BBBLayer(n_hidden, n_outputs, n_weight_samples)
        self.layers = [self.bbb1,
                       th.nn.ReLU(),
                       self.bbb2,
                       th.nn.ReLU()]

    def __call__(self, x: th.Tensor, *args, **kwargs):
        y = x.repeat(n_weight_samples, 1, 1)
        for layer in self.layers:
            y = layer.forward(y)
        return y


class LinearNet(th.nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.l1 = th.nn.Linear(n_inputs, n_hidden)
        self.l2 = th.nn.Linear(n_hidden, n_outputs)
        self.layers = [self.l1,
                       th.nn.ReLU(),
                       self.l2,
                       th.nn.ReLU()]

    def __call__(self, x: th.Tensor, *args, **kwargs):
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == '__main__':
    n_inputs, n_outputs, n_hidden = 2, 3, 10
    n_data_samples, n_weight_samples = 10, 4

    x = th.randn(n_data_samples, n_inputs)
    y = th.randn(n_data_samples, n_outputs)

    model = BBBNet(n_inputs, n_hidden, n_outputs, n_weight_samples)
    # model = LinearNet(n_inputs, n_hidden, n_outputs)

    optimizer = th.optim.Adam(model.parameters())

    n_epochs = 1000
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = bbb_criterion(y_pred, y, [model.bbb1, model.bbb2])
        # loss = th.nn.MSELoss()(y_pred, y)
        loss.backward(retain_graph=(epoch < n_epochs - 1))
        optimizer.step()

        losses.append(loss.detach())
        print(np.array(losses).mean(), epoch)
