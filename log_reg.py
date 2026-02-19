# log_reg.py
import torch
import torch.nn.functional as F

# y is the true label
y = torch.tensor([1.0])
# x is input
x1 = torch.tensor([1.1])
# w is weight, the model's learned parameter
w1 = torch.tensor([2.2])
# b is bias
b = torch.tensor([0.0])
# z is raw output
z = x1 * w1 + b
# a is activation and output, the prediction is a
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
print('activation:', a, 'loss: ' , loss)

import torch.nn.functional as F
from torch.autograd import grad

# y is the true label
y = torch.tensor([1.0])
# x is input
x1 = torch.tensor([1.1])
# w is weight, the model's learned parameter
w1 = torch.tensor([2.2], requires_grad = True)
# b is bias
b = torch.tensor([0.0], requires_grad = True)
# z is raw output
z = x1 * w1 + b
# a is activation and output, the prediction is a
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print('grad_L_w1: ', grad_L_w1)
print('grad_L_b: ', grad_L_b)

loss.backward()
print('w1.grad: ', w1.grad)
print('b.grad: ', b.grad)
