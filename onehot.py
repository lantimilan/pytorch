# onehot.py
# pytorch installation under /home/liyan/python_venv
import torch

def to_onehot(y, num_classes):
    y_onehot = torch.zeros(y.size(0), num_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot

y = torch.tensor([0, 1, 2, 2])

y_enc = to_onehot(y, 3)

print('one-hot encoding:\n', y_enc)

# define a tensor (aka matrix)
Z = torch.tensor([
    [-0.3, -0.5, -0.5],
    [-0.4, -0.1, -0.5],
    [-0.3, -0.94, -0.5],
    [-0.99, -0.88, -0.5]])

print("tensor\n", Z)

# convert tensor to probability distribution via softmax
def softmax(z):
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)).t()

smax = softmax(Z)
print('softmax:\n', smax)
# sum on smax rows should be 1.0

def to_classlabel(z):
    return torch.argmax(z, dim=1)

print('predicted class labels: ', to_classlabel(smax))
print('true class labels: ', to_classlabel(y_enc))

def cross_entropy(softmax, y_target):
    return - torch.sum(torch.log(softmax) * (y_target), dim=1)

# note that each row in Z is a sample, and xent is good
# if it is close to 1.0
xent = cross_entropy(smax, y_enc)
print('Cross Entropy:', xent)

import torch.nn.functional as F

t = F.nll_loss(torch.log(smax), y, reduction='none')
print('nll_loss:\n', t)

t = F.cross_entropy(Z, y, reduction='none')
print('cross_entropy:\n', t)

t = F.cross_entropy(Z, y)
print('avg cross_entropy:\n', t)

t = torch.mean(cross_entropy(smax, y_enc))
print('mean cross_entropy:\n', t)

