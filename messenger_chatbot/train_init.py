import pickle
import random
import numpy as np
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
documents = pickle.load(open('documents.pkl', 'rb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    # initialize bag od words
    bag = []

    # list of tokenized words for the pattern
    pattern_words = doc[0]

    # lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)

training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

n_classes = len(train_y[0])
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics='accuracy')
ml = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)
model.save('chatbot_model.h5', ml)

print('model created')