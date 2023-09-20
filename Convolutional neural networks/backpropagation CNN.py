import tensorflow as tf

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate some dummy data for training
import numpy as np
x_train = np.random.rand(100, 64, 64, 3)
y_train = np.random.randint(0, 10, size=(100,))

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Train the model
model.fit(x_train, y_train, epochs=5)

# Perform backpropagation manually
with tf.GradientTape() as tape:
    # Forward pass
    y_pred = model(x_train, training=True)
    loss_value = tf.keras.losses.categorical_crossentropy(y_train, y_pred)

# Calculate gradients
gradients = tape.gradient(loss_value, model.trainable_variables)

# Update weights using an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
