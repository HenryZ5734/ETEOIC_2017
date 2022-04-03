import logging
import os


def init_log(root):

    # 文件路径判断
    count = 1
    path = os.path.join(root, "results", "train_1")
    while os.path.exists(path):
        count += 1
        path = os.path.join(os.getcwd(), "results", "train_" + str(count))
    os.makedirs(path)

    # 日志初始化
    logging.basicConfig(filename=os.path.join(path, "train.log"), level=logging.INFO)

    # 返回path
    return path
