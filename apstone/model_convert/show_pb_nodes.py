# -- coding: utf-8 --
# @Time : 2023/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone
"""
for tf1.5 model
"""
import tensorflow as tf
from tensorflow.python.framework import graph_util
import argparse

tf.reset_default_graph()  # 重置计算图


def network_structure(args):
    args.model = "model.pb"
    model_path = args.model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        # 获得默认的图
        graph = tf.get_default_graph()
        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            # 得到当前图有几个操作节点
            print("%d ops in the graph." % len(output_graph_def.node))
            op_name = [tensor.name for tensor in output_graph_def.node]
            print(op_name)
            print('=======================================================')
            # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
            summaryWriter = tf.summary.FileWriter('log_graph_' + args.model, graph)
            cnt = 0
            print("%d tensors in the graph." % len(graph.get_operations()))
            for tensor in graph.get_operations():
                # print出tensor的name和值

                print(tensor.name, tensor.values())
                cnt += 1
                if args.n:
                    if cnt == args.n:
                        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help="model name to look")
    parser.add_argument('--n', type=int, default=5,
                        help='the number of first several tensor name to look')  # 当tensor_name过多
    args = parser.parse_args()
    network_structure(args)
