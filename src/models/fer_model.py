import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

class FERModel:
    def __init__(self, data_path, pic_size=48, batch_size=128, epochs=50):
        self.data_path = data_path
        self.pic_size = pic_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.build_model()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def preprocess_fer2013(self):
        """
        Preprocesses the FER-2013 dataset from CSV format to image files.
        """
        # Read the CSV file
        df = pd.read_csv(os.path.join(self.data_path, 'fer2013.csv'))
        
        # Create directories for train and test sets
        for emotion in self.emotion_labels:
            os.makedirs(os.path.join(self.data_path, 'train', emotion), exist_ok=True)
            os.makedirs(os.path.join(self.data_path, 'test', emotion), exist_ok=True)
        
        # Process each image
        for idx, row in df.iterrows():
            pixels = np.array(row['pixels'].split(' '), dtype='float32')
            pixels = pixels.reshape(48, 48)
            
            # Convert to image and save
            plt.imsave(
                os.path.join(
                    self.data_path,
                    'train' if row['Usage'] == 'Training' else 'test',
                    self.emotion_labels[row['emotion']],
                    f'image_{idx}.png'
                ),
                pixels,
                cmap='gray'
            )

    def visualize_data(self):
        """
        Visualizes a sample of training images.
        """
        plt.figure(0, figsize=(12, 20))
        cpt = 0
        for emotion in self.emotion_labels:
            emotion_path = os.path.join(self.data_path, 'train', emotion)
            if os.path.exists(emotion_path):
                for i, img_file in enumerate(os.listdir(emotion_path)[:5]):
                    cpt += 1
                    plt.subplot(7, 5, cpt)
                    img = plt.imread(os.path.join(emotion_path, img_file))
                    plt.imshow(img, cmap="gray")
                    plt.title(emotion)
        plt.tight_layout()
        plt.show()

    def data_generator(self):
        """
        Creates data generators for training and validation datasets.
        """
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=20,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        test_datagen = ImageDataGenerator(rescale=1.0/255)
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(self.pic_size, self.pic_size),
            color_mode="grayscale",
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(self.pic_size, self.pic_size),
            color_mode="grayscale",
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            subset='validation'
        )
        
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_path, 'test'),
            target_size=(self.pic_size, self.pic_size),
            color_mode="grayscale",
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator

    def build_model(self):
        """
        Builds the CNN model architecture.
        """
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(self.pic_size, self.pic_size, 1)))
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
        opt = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, train_generator, validation_generator):
        """
        Trains the model using the training and validation data generators.
        """
        checkpoint = ModelCheckpoint(
            "model_weights.h5",
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        callbacks_list = [checkpoint]
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // train_generator.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.n // validation_generator.batch_size,
            callbacks=callbacks_list
        )
        return history

    def evaluate_model(self, test_generator):
        """
        Evaluates the model on the test set.
        """
        return self.model.evaluate(
            test_generator,
            steps=test_generator.n // test_generator.batch_size
        )

    def save_model(self, path):
        """
        Saves the model architecture and weights to the specified path.
        """
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.h5"))
        print("Saved model to disk")

if __name__ == "__main__":
    # Define paths
    data_path = "data/raw/fer2013"
    
    # Initialize and run the FER model
    fer_model = FERModel(data_path)
    
    # Preprocess the dataset
    fer_model.preprocess_fer2013()
    
    # Visualize data
    fer_model.visualize_data()
    
    # Get data generators
    train_generator, validation_generator, test_generator = fer_model.data_generator()
    
    # Compile and train the model
    fer_model.compile_model()
    history = fer_model.train_model(train_generator, validation_generator)
    
    # Evaluate the model
    test_loss, test_accuracy = fer_model.evaluate_model(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save the model
    fer_model.save_model("models/fer_model")
