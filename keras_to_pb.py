import tensorflow as tf
from tensorflow.python.framework import graph_io

import keras
from keras.models import load_model


def freeze_graph(graph, session, output, save_pb_dir= '.', \
    save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_fronzen = tf.graph_util.convert_variables_to_constants(session, \
            graphdef_inf, output)
        graph_io.write_graph(graphdef_fronzen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_fronzen

#This line must be executed before loading Keras model
keras.backend.set_learning_phase(0)

model_path = "./mymodel.h5"

model = load_model(model_path)
session = keras.backend.get_session()

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs])