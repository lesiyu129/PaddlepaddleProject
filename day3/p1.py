import paddle.dataset
import paddle.dataset.image
import paddle.fluid as fluid
import numpy as np
import paddle
import paddle.reader
import paddle.fluid.layers as layers
import os
import matplotlib.pyplot as plt
from typing import Dict
from PIL import Image
paddle.enable_static()


name_dict = {"apple": 0, "banana": 1, "grape": 2,
             "orange": 3, "pear": 4, }

data_root_path = "./fruits"
test_file_path = "./fruits/test.txt"
train_file_path = "./fruits/train.txt"
name_dict_list: Dict[str, list] = {}


def save_to_dict(file_path, name):
    if name not in name_dict_list:
        img_list = []
        img_list.append(file_path)
        name_dict_list[name] = img_list
    else:
        name_dict_list[name].append(file_path)

# 遍历子目录


for file in os.listdir(data_root_path):
    sub_dir = data_root_path + "/" + file
    if os.path.isdir(sub_dir):
        for img in os.listdir(sub_dir):
            img_path = sub_dir + "/" + img
            save_to_dict(img_path, file)


with open(train_file_path, "w") as f:
    pass
with open(test_file_path, "w") as f:
    pass

for name, image_list in name_dict_list.items():
    i = 0
    num = len(image_list)
    print(name, ":", num)

    for img in image_list:
        line = "%s\t%d\n" % (img, name_dict[name])
        if i % 10 == 0:
            with open(test_file_path, "a") as f:
                f.write(line)
        else:
            with open(train_file_path, "a") as f:
                f.write(line)
        i += 1

print("数据预处理完成")


def train_mapper(sample):
    img_path, label = sample
    if not os.path.exists(img_path):
        return None
    im = paddle.dataset.image.load_image(img_path)
    im = paddle.dataset.image.simple_transform(
        im, 128, 128, is_color=True, is_train=True)  # 64*64
    im = im.astype('float32') / 255.0
    return im, label


def train_r(train_list, buffer_size=1024):
    def reader():
        with open(train_list, "r") as f:
            for line in f.readlines():
                img_path, label = line.strip().split("\t")
                yield img_path, int(label)

    return paddle.reader.xmap_readers(train_mapper, reader, os.cpu_count(), buffer_size)


def CNN_model(img, type_size):
    # 卷积/激活/池化/dropout
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,  # 输入
        num_filters=32,  # 卷积核数量
        filter_size=3,  # 卷积核大小
        pool_size=2,  # 池化窗口大小
        pool_stride=2,  # 池化步长
        act="relu")  # 激活函数

    drop = fluid.layers.dropout(
        x=conv_pool_1, dropout_prob=0.5)  # 丢弃率0.5，防止过拟合

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=drop,
        num_filters=64,
        filter_size=3,
        pool_size=2,
        pool_stride=2,
        act="relu")

    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)

    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=drop,
        num_filters=128,
        filter_size=3,
        pool_size=2,
        pool_stride=2,
        act="relu")

    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)

    fc = fluid.layers.fc(input=drop, size=512, act="relu")
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)

    # 输出层
    predict = fluid.layers.fc(input=drop, size=type_size, act="softmax")

    return predict


batch_size = 32
train_data = train_r(train_file_path)
train_reader = paddle.reader.shuffle(train_data, 1024)
train_batch = paddle.batch(train_reader, batch_size=batch_size)

# 定义训练过程
img = fluid.layers.data(name='img', shape=[3, 128, 128], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


predict = CNN_model(img, len(name_dict))
# 定义损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=predict, label=label)  # 交叉熵损失函数
avg_cost = fluid.layers.mean(x=cost)  # 平均损失
optimizer = fluid.optimizer.Adam(learning_rate=0.001)  # 优化器
optimizer.minimize(avg_cost)  # 指定优化器

accuracy = fluid.layers.accuracy(input=predict, label=label)  # 准确率

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feeder = fluid.DataFeeder(place=place, feed_list=[img, label])

costs = []
accs = []
times = 0
batches = []

for epoch in range(10):
    for batch_id, data in enumerate(train_batch()):
        times += 1
        train_cost, train_acc = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, accuracy])
        if batch_id % 10 == 0:
            print("epoch:%d, batch:%d, cost:%.5f, acc:%.5f"
                  % (epoch, batch_id, train_cost[0], train_acc[0]))
            costs.append(train_cost[0])
            accs.append(train_acc[0])
            batches.append(times)

# 保存模型
model_save_dir = './model'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

fluid.io.save_inference_model(model_save_dir, ['img'], [predict], exe)

print("模型保存成功")
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=20)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batches, costs, color='red', label='cost')
plt.plot(batches, accs, color='blue', label='acc')
plt.legend()
plt.grid()
plt.savefig('./train.png')
plt.show()

# 测试
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
model_save_dir = "./model"  # 模型保存路径

# 加载数据


def load_img(path):
    img = paddle.dataset.image.load_and_transform(
        path, 128, 128, False).astype("float32")
    img = img / 255.0
    return img


infer_imgs = []  # 存放要预测图像数据
test_img = "./grape_1.png"  # 待预测图片
infer_imgs.append(load_img(test_img))  # 加载图片，并且将图片数据添加到待预测列表
infer_imgs = np.array(infer_imgs)  # 转换成数组

# 加载模型
infer_program, feed_target_names, fetch_targets = \
    fluid.io.load_inference_model(model_save_dir, infer_exe)
# 执行预测
results = infer_exe.run(infer_program,  # 执行预测program
                        feed={feed_target_names[0]: infer_imgs},  # 传入待预测图像数据
                        fetch_list=fetch_targets)  # 返回结果
print(results)

result = np.argmax(results[0])  # 取出预测结果中概率最大的元素索引值
for k, v in name_dict.items():  # 将类别由数字转换为名称
    if result == v:  # 如果预测结果等于v, 打印出名称
        print("预测结果:", k)  # 打印出名称

# 显示待预测的图片
img = Image.open(test_img)
plt.imshow(img)
plt.show()
