import torch

class LinearLayerModel(torch.nn.Module):
    """
    This class represents a simple one-layer linear model
    with a log softmax activation function.
    """
    def __init__(self, input_size: int, output_size: int, activation_function: torch.nn.Module = torch.nn.LogSoftmax(dim=1)):
        super(LinearLayerModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_function(self.linear(x))
    
    def predict(self, output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.activation_function(output), dim=1)
