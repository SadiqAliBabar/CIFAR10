import torch
from model.cnn import CNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def test_model(testloader):
    model = CNN()
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')
