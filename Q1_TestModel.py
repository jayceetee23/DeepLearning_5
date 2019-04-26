import pandas as pd
import re
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load created model model.h5
model = load_model('model.h5')

# Categeories for 'sentiment'
categories = ['Positive', 'Negative']

# Test text to run through model
text = ["A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump"]

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(text)
X = tokenizer.texts_to_sequences(text)
# print(X)
X = pad_sequences(X,maxlen=28)
# print(X)

# Make prediction on model
prediction = model.predict(X)
print(prediction)

import numpy as np

print(np.argmax(prediction))

pred_name = categories[np.argmax(prediction)]
print(pred_name)