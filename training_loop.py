# training_loop.py
import torch

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5],
    ])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
    ])
y_test = torch.tensor([0, 1])

# Prepare dataset
from torch.utils.data import Dataset

class ToyDataset (Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        # retrieve one data record and one label
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# expects 5 records in training dataset
# note that len is defined in __len__
print('expects 5 records, got', len(train_ds))

# Sample data using DataLoader
from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
        dataset = train_ds,
        batch_size = 2,
        shuffle = True,
        num_workers = 0)

test_loader = DataLoader(
        dataset = test_ds,
        batch_size = 2,
        shuffle = False,
        num_workers = 0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx + 1}:\n", x, "\n", y)
print('iterate one more time to observe shuffle order change')
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx + 1}:\n", x, "\n", y)

train_loader = DataLoader(
        dataset = train_ds,
        batch_size = 2,
        shuffle = True,
        num_workers = 0,
        drop_last = True)
for idx, (x, y) in enumerate(train_loader):
    print(f"with drop_last, Batch {idx + 1}:\n", x, "\n", y)

import torch.nn.functional as F

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

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(
        model.parameters(), lr=0.5
        )

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### logging
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f"  | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f"  | Train loss: {loss:.2f}")
    model.eval()

    # insert optional model evaluation code
num_params = sum(p.numel() 
                 for p in model.parameters() 
                 if p.requires_grad)
print(num_params)

# prediction
model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

predictions = torch.argmax(probas, dim=1)
print(predictions)

s = torch.sum(predictions == y_train)
print(s)

def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

print(compute_accuracy(model, train_loader))
print(compute_accuracy(model, test_loader))

