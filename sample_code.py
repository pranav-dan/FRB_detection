import tensorflow as tf
import keras
model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(3,)),
    keras.layers.Softmax()])
config = model.to_json()
loaded_model = keras.models.model_from_json(config)