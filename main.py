import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from cnn import CNN
from trainer import Trainer
from evaluator import Evaluator

def train_model(dataset_size, epochs, file_name):
    field_path = "assets/training_field_1.png"
    generator = Generator(field_path)
    model = CNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, criterion, optimizer, device)
    images, angles = generator.generate_dataset(dataset_size)
    train_images, val_images, train_angles, val_angles = trainer.data_split(images, angles)
    train_loader = trainer.loader(train_images, train_angles)
    val_loader = trainer.loader(val_images, val_angles)
    trainer.train(train_loader, val_loader, epochs)
    torch.save(model.state_dict(), file_name + '.pth' )

def eval_model(file_name, field, num_images):
    model = CNN()
    state_dict = file_name + '.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    field_path = "assets/testing_field_" + field + ".png"
    generator = Generator(field_path)
    evaluator = Evaluator(model, state_dict, device, generator)
    evaluator.test_all_angles()
    evaluator.test_randomly_generated_images(num_images)
    evaluator.create_gif()

if __name__ == "__main__":
    mode = input("Do you want to train (type 'train') or evaluate (type 'eval') a model? ")
    if mode == 'train':
        dataset_size = int(input("How many images do you want to train the model on? "))
        epochs = int(input("How many epochs do you want to train the model for? "))
        file_name = input("Please specify a name for the model: ")
        train_model(dataset_size, epochs, file_name)   
        print("Training of model " + file_name + " complete!")
    elif mode == 'eval':
        file_name = input("Please specify the name of the model to be evaluated: ")
        field = input("Which testing field ('1' or '2') should be used? ")
        num_images = int(input("Please specify how many images should be included in the gif output: "))
        eval_model(file_name, field, num_images)
        print("Evaluation of model " + file_name + " complete!")