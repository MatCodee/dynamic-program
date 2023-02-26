'''
Este es le modelo de la red neironal donde vamos a ejecutar los parametros necesarios
para que el modelo genere un input de exito
'''
from tensorflow.keras.layers import Input

input_shape = (None, None, 3)  # Formato de imagen RGB
input_a = Input(shape=input_shape)  # Video sin editar
input_b = Input(shape=input_shape)  # Video editado de referencia

input_data = (input_a, input_b)
