import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Create random input tensors for training and testing
train_data = torch.randn((64, 3, 32, 32))
train_labels = torch.randint(0, 10, (64,))
test_data = torch.randn((16, 3, 32, 32))

# Normalize the input data
train_data = (train_data - torch.mean(train_data)) / torch.std(train_data)
test_data = (test_data - torch.mean(test_data)) / torch.std(test_data)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the CNN model
model = CNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(5):
    running_loss = 0.0
    for i in range(len(train_data)):
        inputs, labels = train_data[i].unsqueeze(0).to(device), train_labels[i].unsqueeze(0).to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished training')

# Test the model
with torch.no_grad():
    model.eval()
    total = 0
    correct = 0
    for i in range(len(test_data)):
        inputs = test_data[i].unsqueeze(0).to(device)
        labels = torch.randint(0, 10, (1,)).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy: %.2f %%' % accuracy)

