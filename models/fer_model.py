import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

class FERModel:
    def __init__(self, base_path, pic_size=48, batch_size=128, epochs=50):
        self.base_path = base_path
        self.pic_size = pic_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.build_model()
    
    def visualize_data(self):
        """
        Visualizes a sample of training images.
        """
        plt.figure(0, figsize=(12, 20))
        cpt = 0
        for expression in os.listdir(self.base_path + "train"):
            for i in range(1, 6):
                cpt += 1
                plt.subplot(7, 5, cpt)
                img = load_img(self.base_path + "train/" + expression + "/" + os.listdir(self.base_path + "train/" + expression)[i], target_size=(self.pic_size, self.pic_size))
                plt.imshow(img, cmap="gray")
        plt.tight_layout()
        plt.show()

    def data_generator(self):
        """
        Creates data generators for training and validation datasets.
        """
        train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, rotation_range=20, horizontal_flip=True)
        validation_datagen = ImageDataGenerator(rescale=1.0/255)
        train_generator = train_datagen.flow_from_directory(self.base_path + "train", target_size=(56, 56), color_mode="grayscale", batch_size=self.batch_size, class_mode='categorical', shuffle=True)
        validation_generator = validation_datagen.flow_from_directory(self.base_path + "validation", target_size=(56, 56), color_mode="grayscale", batch_size=self.batch_size, class_mode='categorical', shuffle=False)
        return train_generator, validation_generator

    def build_model(self):
        """
        Builds the CNN model architecture.
        """
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(56, 56, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(7, activation='softmax'))
        return model

    def compile_model(self):
        """
        Compiles the model with the Adam optimizer and categorical crossentropy loss.
        """
        opt = Adam(lr=0.0001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, train_generator, validation_generator):
        """
        Trains the model using the training and validation data generators.
        """
        # Define a checkpoint to save the best model
        checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        # Train the model
        history = self.model.fit_generator(generator=train_generator, steps_per_epoch=train_generator.n // train_generator.batch_size, epochs=self.epochs, validation_data=validation_generator, validation_steps=validation_generator.n // validation_generator.batch_size, callbacks=callbacks_list)
        return history

    def save_model(self, path):
        """
        Saves the model architecture and weights to the specified path.
        """
        self.model.save(path)
        model_json = self.model.to_json()
        with open(path + "/model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(path + "/model.h5")
        print("Saved model to disk")

if __name__ == "__main__":
    # Define paths
    base_path = "/content/img/images/"
    
    # Initialize and run the FER model
    fer_model = FERModel(base_path)
    fer_model.visualize_data()
    train_generator, validation_generator = fer_model.data_generator()
    fer_model.compile_model()
    fer_model.train_model(train_generator, validation_generator)
    fer_model.save_model("/content")
