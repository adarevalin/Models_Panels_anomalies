import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l1_l2
from cargar_datos import obtener_generadores

# Ruta del archivo JSON y la carpeta de imágenes
path_file = 'C:/Users/USUARIO/Downloads/data_solar/InfraredSolarModules/module_metadata.json'
image_folder = 'C:/Users/USUARIO/Downloads/data_solar/InfraredSolarModules'

# Obtener generadores de datos
train_generator, test_generator = obtener_generadores(path_file, image_folder)

# Parámetros de regularización L1 y L2
l1_reg = 0.0001
l2_reg = 0.0001

# Crear el modelo
model = Sequential()

model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu', input_shape=(40, 24, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.05))

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.05))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.05))

model.add(Flatten())
model.add(Dense(units=120, activation='relu',kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
model.add(Dropout(0.5))  # Capa de Dropout para regularización

model.add(Dense(units=12, activation='softmax'))

model.summary()

# Compilar el modelo con una tasa de aprendizaje inicial más baja
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



# Entrenar el modelo usando los generadores de datos
model.fit(train_generator,
          epochs=500,
          validation_data=test_generator)

# Evaluar el modelo
loss, accuracy = model.evaluate(test_generator)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')


# Guardar el modelo en formato .h5
model.save('modelo.h5')
print("Modelo guardado en formato .h5.")

import pickle
# Guardar el modelo de TensorFlow/Keras en formato .pkl (solo los parámetros)
# Nota: Esta no es la forma estándar de guardar un modelo de Keras en .pkl,
# pero puedes guardar los parámetros que desees si es necesario.
with open('model_params.pkl', 'wb') as file:
    params = model.get_weights()  # Obtener los pesos del modelo como ejemplo
    pickle.dump(params, file)
print("Parámetros del modelo guardados en formato .pkl.")