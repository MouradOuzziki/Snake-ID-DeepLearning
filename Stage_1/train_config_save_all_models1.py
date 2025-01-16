import time
import timm
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, Adamax
from torch.optim import SGD, lr_scheduler
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

# number of classes
N_CLASSES = 772

# Load pre-trained EfficientNet B0 from timm
model_efficientnet_b0 = timm.create_model('efficientnet_b0', pretrained=True)
model_vgg16 = timm.create_model('vgg16', pretrained=True)
model_vgg19 = timm.create_model('vgg19', pretrained=True)


# Modify the final classification layer (classifier) for 772 classes
model_efficientnet_b0.classifier = nn.Linear(in_features=model_efficientnet_b0.classifier.in_features, out_features=N_CLASSES)
model_vgg16.head.fc = nn.Linear(in_features=model_vgg16.head.fc.in_features, out_features=N_CLASSES)
model_vgg19.head.fc = nn.Linear(in_features=model_vgg19.head.fc.in_features, out_features=N_CLASSES)


# Check if GPU is available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Print the device being used
print(f"Using device: {device}")

# Move the model to GPU if possible
model_efficientnet_b0.to(device)
model_vgg16.to(device)
model_vgg19.to(device)


lr = 0.001
criterion = nn.CrossEntropyLoss()


# Define the function for training and validation of the model
def train__model(model, train_loader, valid_loader, num_epochs, initial_epochs):

    optimizer = Adamax(model.parameters(), lr=lr)
    ##optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5)

    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []

    # Training loop
    for epoch in range(initial_epochs, initial_epochs + num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        avg_train_loss = 0.
        train_preds = []
        train_labels = []

        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            y_preds = model(images)
            loss = criterion(y_preds, labels)
            avg_train_loss += loss.item() / len(train_loader)

            # Store predictions and labels for metrics
            preds = y_preds.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            loss.backward()
            optimizer.step()

        # Calculate training accuracy and F1 score
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        # Validation phase
        model.eval()
        avg_val_loss = 0.
        val_preds = []
        val_labels = []

        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                y_preds = model(images)

            preds = y_preds.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

            loss = criterion(y_preds, labels)
            avg_val_loss += loss.item() / len(valid_loader)

        # Calculate validation accuracy and F1 score
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        # Update learning rate scheduler
        scheduler.step()

        # Print epoch summary
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}/{initial_epochs + num_epochs} - train_loss: {avg_train_loss:.4f} val_loss: {avg_val_loss:.4f} '
              f'Train F1: {train_f1:.6f} Val F1: {val_f1:.6f} Train Acc: {train_accuracy:.6f} '
              f'Val Acc: {val_accuracy:.6f} time: {elapsed:.0f}s')

        # Append metrics to lists
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

    # Return metrics for plotting
    return train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, initial_epochs + num_epochs

# Define a function to plot loss and accuracy
def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, model_name):
    plt.figure(figsize=(18, 8))

    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy - {model_name}')
    plt.legend()

    # Plot training and validation F1 score
    plt.subplot(1, 3, 3)
    plt.plot(train_f1_scores, label='Training F1 Score')
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'Training and Validation F1 Score - {model_name}')
    plt.legend()

    plt.show()
    plt.savefig(f"{model_name}_snakeCLEF2021.png")

# List of models
models = [model_efficientnet_b0, model_vgg16, model_vgg19]
initial_epochs = 10 

#(train_loader, valid_loader) from Data_prepartion1.py


# train each model
for model in models:
    print(f"Training {model.__class__.__name__}...")
    
    # Get the number of epochs to train for
    add_epochs = input(f"How many epochs do you want to train {model.__class__.__name__}? (Current epoch count: {initial_epochs}): ").strip()
    
    # Check if user wants to add more epochs
    while not add_epochs.isdigit():
        add_epochs = input("Invalid input. Please enter a number of epochs: ").strip()

    add_epochs = int(add_epochs)
    
    train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, initial_epochs = train__model(
        model, train_loader, valid_loader, num_epochs=10, initial_epochs=initial_epochs)

    # Save the model after training in the "models" directory
    model_save_path = f"../Snake-ID-DeepLearning/models/{model.__class__.__name__}_stage1.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")

    # After training, plot loss and accuracy for each model
    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, model.__class__.__name__)