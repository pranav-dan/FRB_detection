#!/usr/bin/env python3

import glob
import logging
import os
import string
import pdb
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file 
from tensorflow.keras.layers import Average , Dense 


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
PATH_TO_WEIGHTS = "http://psrpop.phys.wvu.edu/download.php?val="

logger = logging.getLogger(__name__)


def open_n_layers_for_training(model, nlayers):
    """
    Makes nlayers of the model trainable.
    nlayers start from the top of the model.
    Top (or head) refers to the classification layer. The opening of layers for training starts from top.

    :param model: Model to open layers of
    :type model: Model
    :param nlayers: Number of (trainable) layers to open.
    :type nlayers: int
    :return: model
    """
    mask = np.zeros(len(model.layers), dtype=np.bool_)
    mask[-nlayers:] = True
    for layer, mask_val in zip(model.layers, mask):
        layer.trainable = mask_val
    return model


def ready_for_train(model, nf, ndt, nft):
    """
    This makes the model ready for training, it opens the layers for training and complies it.

    :param model: model to train
    :type: Model
    :param nf: Number of layers to train post FT and DT models
    :type nf: int
    :param ndt: Number of layers in DT model to train
    :type ndt: int
    :param nft: Number of layers in FT model to train
    :type nft : int
    :return: compiled model ready for training
    """

    # Make all layers non trainable first
    model.trainable = False
    model = open_n_layers_for_training(model, nf)

    # Get the FT and DT models to open them up for training
    model.layers[4] = open_n_layers_for_training(model.layers[4], nft)
    model.layers[5] = open_n_layers_for_training(model.layers[5], ndt)

    model_trainable = Model(model.inputs, model.outputs)

    # Adam optimizer with imagenet defaults
    optimizer = Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    )

    # Compile
    model_trainable.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    return model_trainable

def ready_for_train_ensemble(models, nf, ndt, nft):
    """
    This makes the model ready for training, it opens the layers for training and complies it.

    :param model: model to train
    :type: Model
    :param nf: Number of layers to train post FT and DT models
    :type nf: int
    :param ndt: Number of layers in DT model to train
    :type ndt: int
    :param nft: Number of layers in FT model to train
    :type nft : int
    :return: compiled model ready for training
    """

    # Make all layers non trainable first
    for model in models:
        model.trainable = False
        model = open_n_layers_for_training(model, nf)

        # Get the FT and DT models to open them up for training
        model.layers[4] = open_n_layers_for_training(model.layers[4], nft)
        model.layers[5] = open_n_layers_for_training(model.layers[5], ndt)
    
    outputs = [model(model.input) for model in models] 
    inputs = models[0].inputs
    avg_output = Average()(outputs) 
    num_units = avg_output.shape[-1]

# Create a dense layer with the same number of units as the average layer output
    dense_output = Dense(num_units, activation='softmax')(avg_output)

    print("avg layer:", dense_output)
    print("model output:", models[0].outputs)

    model_trainable = Model(inputs,[dense_output])
    # Adam optimizer with imagenet defaults
    optimizer = Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    )

    # Compile
    model_trainable.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    return model_trainable
##ValueError: Graph disconnected: cannot obtain value for tensor KerasTensor(type_spec=TensorSpec(shape=(None, 256, 256, 1), 
# dtype=tf.float32, name='data_freq_time'), name='data_freq_time', description="created by layer 'data_freq_time'") at layer "model_3". 
# The following previous layers were accessed without issue: ['model_3']

def get_model(model_idx, model_json,model_csv_path):
    """

    :param model_idx: model string between a--j
    :type model_idx: str
    :return: Model
    """
    # Get the model from the folder
    logger.info(f"Getting model {model_idx}")
    path = os.path.split(__file__)[0]
    # model_json = glob.glob(f"{path}/models/{model_idx}_FT*/*json")[0]
    # print(model_json)
    # Read the model from the json
    # pdb.set_trace()
    with open(model_json, "r") as j:
        model_json_content = j.read()
    # pdb.set_trace()
    model = tf.keras.models.model_from_json(model_json_content)

    # get the model weights, if not present download them.
    # model_list = pd.read_csv(f"{path}/models/model_list.csv")
    model_list= pd.read_csv(model_csv_path)
    model_index = string.ascii_lowercase.index(model_idx)

    weights = get_file(
        model_list["model"][model_index],
        PATH_TO_WEIGHTS + model_list["model"][model_index],
        file_hash=model_list["hash"][model_index],
        cache_subdir="models",
        hash_algorithm="md5",
    )

    # dump weights
    model.load_weights(weights)
    return model
