from src import LinearLayerModel, train, SGD, accuracy, confusion_matrix, BasicCNN
from torch.optim import Adam
import matplotlib.pyplot as plt

input_size = 784  # for Fashion MNIST
output_size = 10  # for Fashion MNIST
batch_size = 1000
epochs = 10

#model = LinearLayerModel(input_size, output_size)
#
#print("Training with custom SGD...")
#custom_optimizer_sgd = SGD(model.parameters(), lr=0.1)
#SGD_losses,SGD_preds, SGD_labels =train(model, custom_optimizer_sgd, batch_size, epochs)
#
#print("Training with Adam...")
#model = LinearLayerModel(input_size, output_size)
#optimizer_adam = Adam(model.parameters(), lr=0.01)
#Adam_losses, Adam_preds, Adam_labels = train(model, optimizer_adam, batch_size=100, epochs=10)
#
## Plot the losses
#plt.plot(SGD_losses, label="SGD")
#plt.plot(Adam_losses, label="Adam")
#plt.title("Loss vs Epochs for Fashion MNIST dataset")
#plt.xlabel("Epoch")
#plt.ylabel("Loss")
#plt.legend()
#plt.show()
#
## Plot the confusion matrix
#print(f"Accuracy with custom SGD: {accuracy(SGD_preds, SGD_labels)}")
#print(f"Accuracy with Adam: {accuracy(Adam_preds, Adam_labels)}")
#SGD_conf_matrix = confusion_matrix(SGD_preds, SGD_labels)
#Adam_conf_matrix = confusion_matrix(Adam_preds, Adam_labels)
#plt.imshow(SGD_conf_matrix, cmap="hot", interpolation="nearest")
#plt.title("Confusion matrix for custom SGD")
#plt.show()
#plt.imshow(Adam_conf_matrix, cmap="hot", interpolation="nearest")
#plt.title("Confusion matrix for Adam")
#plt.show()


model = BasicCNN(output_size)
print("Training CNN with custom SGD...")
custom_optimizer_sgd = SGD(model.parameters(), lr=0.1)
SGD_losses, SGD_preds, SGD_labels = train(model, custom_optimizer_sgd, batch_size, epochs)

print("Training CNN with Adam...")
optimizer_adam = Adam(model.parameters(), lr=0.01)
Adam_losses, Adam_preds, Adam_labels = train(model, optimizer_adam, batch_size=100, epochs=10)

# Plot the losses
plt.plot(SGD_losses, label="SGD")
plt.plot(Adam_losses, label="Adam")
plt.title("Loss vs Epochs for Fashion MNIST dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot the confusion matrix
print(f"Accuracy with custom SGD: {accuracy(SGD_preds, SGD_labels)}")
print(f"Accuracy with Adam: {accuracy(Adam_preds, Adam_labels)}")
SGD_conf_matrix = confusion_matrix(SGD_preds, SGD_labels)
Adam_conf_matrix = confusion_matrix(Adam_preds, Adam_labels)
plt.imshow(SGD_conf_matrix, cmap="hot", interpolation="nearest")
plt.title("Confusion matrix for custom SGD")
plt.show()
plt.imshow(Adam_conf_matrix, cmap="hot", interpolation="nearest")
plt.title("Confusion matrix for Adam")
plt.show()

