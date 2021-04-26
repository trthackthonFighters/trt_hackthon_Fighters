from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

import configure_finetuning
from finetune import preprocessing
from finetune import task_builder
from model import modeling
from model import optimization
from util import training_utils
from util import utils

from datetime import datetime
import numpy as np
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import onnxruntime as ort
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import tensorflow as tf
import tf2onnx
from tf2onnx.tfonnx import process_tf_graph, tf_optimize
import argparse
import time

def main():
    tf.set_random_seed(1234)
    np.random.seed(0)
    batch_size = 1
    tf_datatype = tf.int32
    np_datatype = np.int32
    iterations = 10

    features_ph = {}
    features_ph["input_ids"] = tf.placeholder(dtype=tf_datatype,
                                                shape=[batch_size, 128],
                                                name="input_ids")
    features_ph["input_mask"] = tf.placeholder(dtype=tf_datatype,
                                                shape=[batch_size, 128],
                                                name="input_mask")
    features_ph["token_type_ids"] = tf.placeholder(dtype=tf_datatype,
                                                    shape=[batch_size, 128],
                                                    name="token_type_ids")

    features_data = {}
    features_data["input_ids"] = np.random.rand(batch_size,
                                                128).astype(np_datatype)
    features_data["input_mask"] = np.random.rand(batch_size,
                                                    128).astype(np_datatype)
    features_data["token_type_ids"] = np.random.rand(
        batch_size, 128).astype(np_datatype)

    features_feed_dict = {
        features_ph[key]: features_data[key]
        for key in features_ph
    }

    finetuning_config = configure_finetuning.FinetuningConfig("ConvBert", "./")
    bert_config = training_utils.get_bert_config(finetuning_config)
    bert_model = modeling.BertModel(
        bert_config=bert_config,
        is_training=False,
        input_ids=features_ph["input_ids"],
        input_mask=features_ph["input_mask"],
        token_type_ids=features_ph["token_type_ids"])

    #outputs_names = "discriminator_predictions/Sigmoid:0,discriminator_predictions/truediv:0,discriminator_predictions/Cast_2:0,discriminator_predictions/truediv_1:0"
    graph_outputs = bert_model.get_sequence_output()
    outputs_names = graph_outputs.name
    print("graph output: ", graph_outputs)
    run_op_list = []
    outputs_names_with_port = outputs_names.split(",")
    outputs_names_without_port = [ name.split(":")[0] for name in outputs_names_with_port ]
    for index in range(len(outputs_names_without_port)):
        run_op_list.append(outputs_names_without_port[index])
    inputs_names_with_port = [features_ph[key].name for key in features_ph]

    # define saver
    #saver = tf.train.Saver(var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            sess.run(run_op_list, feed_dict=features_feed_dict)
        tf_time_sum = 0
        a = datetime.now()
        for i in range(iterations):
            tf_result = sess.run(run_op_list, feed_dict=features_feed_dict)
        b = datetime.now()
        tf_time_sum = (b - a).total_seconds()
        tf_time = "[INFO] TF  execution time: " + str(
            tf_time_sum * 1000 / iterations) + " ms"
        # tf_result = tf_result.flatten()

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, outputs_names_without_port)
        # frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
        # save frozen model
        with open("ConvBert.pb", "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())

    # tf.reset_default_graph()
    # tf.import_graph_def(frozen_graph, name='')

    # #with tf.Session(config=config) as sess:
    # sess = tf.Session(config=config)
    # graph_def = tf_optimize(inputs_names_with_port, outputs_names_without_port,
    #                         sess.graph_def, True)

    # with open("ConvBert_optimized_model.pb", "wb") as ofile:
    #     ofile.write(graph_def.SerializeToString())

    onnx_model_file = "ConvBert.onnx"
    command = "python3 -m tf2onnx.convert --input ConvBert.pb --output %s --fold_const --opset 12 --verbose" % onnx_model_file
    command += " --inputs "
    for name in inputs_names_with_port:
        command += "%s," % name
    command = command[:-1] + " --outputs "
    for name in outputs_names_with_port:
        command += "%s," % name
    command = command[:-1]
    os.system(command)
    print(command)
    #exit(0)

    command = "trtexec - -onnx = ConvBert.onnx - -verbose"
    os.system(command)
    print(command)


if "__main__" ==  __name__:
    main()