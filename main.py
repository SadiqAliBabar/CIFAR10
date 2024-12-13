import torch
from utils.data_loader import load_data
from utils.train import train_model
from utils.test import test_model

def main():
    trainloader, testloader = load_data(batch_size=64)
    train_model(trainloader, epochs=5)
    test_model(testloader)

if __name__ == "__main__":
    main()
