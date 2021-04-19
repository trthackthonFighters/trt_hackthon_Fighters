
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

sess = tf.Session()
Saver = tf.train.import_meta_graph("convbert_medium-small/model.ckpt.meta")
Saver.restore(sess, tf.train.latest_checkpoint("convbert_medium-small/"))
graph = tf.get_default_graph()

output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['name'])

with tf.gfile.FastGFile('convbert.pb', mode='wb') as f:
    f.write(output_graph_def.SerializeToString())


