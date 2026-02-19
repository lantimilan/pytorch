# perceptron.py
import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
                # 1st hidden layer
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),

                # 2nd layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),

                # output layer
                torch.nn.Linear(20, num_outputs),
                )

    def forward(self, x):
        logits = self.layers(x)
        return logits # outputs of the last layer

model = NeuralNetwork(50, 3)
# see the model structure
print(model)

num_params = sum(p.numel() 
                 for p in model.parameters() 
                 if p.requires_grad)
print("Total number of trainable model parameters:", 
      num_params)

print(model.layers[0].weight)
print(model.layers[0].weight.shape)
print(model.layers[0].bias)
print(model.layers[0].bias.shape)

torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)

torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
print(out)

# inference does not need backprop
with torch.no_grad():
    out = model(X)
print(out)

# compute class label with softmax
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)
