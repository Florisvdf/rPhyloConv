import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential

# Creating a keras model
def build_model(input_shape, n_layers, filters, kernel_size, pool_size, activation, n_classes, dropout, learning_rate, loss):
    rows, cols = input_shape
    layers = []
    for i in range(n_layers - 1):
        layers.append(tf.keras.layers.Conv2D(filters = filters, 
                                          kernel_size = kernel_size,
                                          activation = activation))
        layers.append(tf.keras.layers.MaxPooling2D(pool_size = pool_size))
    layers.append(tf.keras.layers.Conv2D(filters = 1,
                                      kernel_size = (1, 1),
                                      activation = activation))
    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(32, activation='relu'))
    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(n_classes, activation='softmax'))
    model = Sequential(layers)
    model.build((None, rows, cols, 1))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = opt, 
                  loss = loss 
                  #metrics = [auroc]
                  )
    session = tf.compat.v1.keras.backend.get_session()
    # Reinitializing model everytime build_model is called
    for layer in model.layers:
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer
        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it
            var = getattr(init_container, key.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))
    return model