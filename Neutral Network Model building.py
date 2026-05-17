from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create model
model = Sequential()

# Hidden layer
model.add(Dense(16,
                input_dim=4,
                activation='relu'))

# Output layer
model.add(Dense(1,
                activation='sigmoid'))   # use 'linear' for regression

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',   # use 'mse' for regression
    metrics=['accuracy']
)

# Model summary
model.summary()
