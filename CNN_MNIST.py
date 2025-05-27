
#CNN Architecture 

#classifying mnist dataset

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.utils import to_categorical

#Loading MNIST dataset :
    
from keras.datasets import mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()

#Now that we have loaded the data, we have to reshape this to whatever dimensions we need

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')

#x_train.shape returns the dimensions of the image
#if it was a tabular data, it would give us rowsxcolumns
#for images , it will give us the shape of image

X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')



#Now we have to normalise the pixel values between 0 and 1 so we divide by 255

X_train=X_train/255
X_test=X_test/255


#now we do one-hot encoding 
#a neural network often expects its categorical outputs
#to be in one-hot encoded format for classification tasks
#this is only for training data.

y_train=to_categorical(y_train) #one_hot encoding 
y_test=to_categorical(y_test)


num_classes=y_test.shape[1]
#this basically gives us the number of labels in the dataset



#main conv block :
    
def cnn():
    model=Sequential()
    model.add(Input(shape=(28,28,1)))
    model.add(Conv2D(16,(2,2),strides=(1,1),activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(32,(2,2),activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    model.add(Dense(100,activation = 'relu'))
    model.add(Dense(num_classes,activation='softmax'))
    
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=cnn()
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=1024,verbose=1)
scores=model.evaluate(X_test,y_test,verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))
















