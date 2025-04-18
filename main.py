import tensorflow as tf
import matplotlib.pyplot as plt
from model import model
print(tf.config.list_physical_devices('GPU'))



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
random_model = model.create_best_model_skips(10) # load model

tf.keras.utils.plot_model(random_model,"model.png")

print(x_train.shape)
print(y_train.shape)
print(y_train[:10])


x_train = x_train/255.0
x_test = x_test/255.0

history = random_model.fit(x_train, y_train, epochs=35, batch_size=64, validation_split=0.1, verbose=1)

print("Plotting the model accuracy against the epoch")


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()



