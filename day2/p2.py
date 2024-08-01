import os
import numpy as np
import paddle
import paddle.dataset
import paddle.dataset.uci_housing
import paddle.reader
import paddle.fluid as fluid
import matplotlib.pyplot as plt
paddle.enable_static()


# 1. 准备数据
BUF_SIZE = 500  # 打乱样本数量
BATCH_SIZE = 32  # 每次训练样本数量

# 2. 定义模型、损失函数、优化器
reader = paddle.dataset.uci_housing.train()
random_reader = paddle.reader.shuffle(reader, buf_size=BUF_SIZE)
batch_train_reader = paddle.batch(
    random_reader, batch_size=BATCH_SIZE, drop_last=True)

# for sample in batch_train_reader():
#     print(sample)
#     break

x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

y_predict = fluid.layers.fc(input=x,  # 输入（13个特征）
                            size=1,  # 输出值个数（神经元数量）
                            act=None)  # 回归问题不使用激活函数
# 损失函数
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
# 求均值
avg_cost = fluid.layers.mean(cost)
# 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.001)
# 训练
optimizer.minimize(avg_cost)

# 3. 训练模型
place = fluid.CPUPlace()  # CPU
exe = fluid.Executor(place)  # 创建执行器
exe.run(fluid.default_startup_program())
# feeder: 数据喂入器（将reader读取到的数据，拼成字典格式输入到模型）
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])  # 设置喂入x/y张量

iter = 0
iters = []
train_costs = []

EPOCH_NUM = 120
for pass_id in range(EPOCH_NUM):
    for batch_id, data in enumerate(batch_train_reader()):
        train_cost, = exe.run(
            fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avg_cost])
        if batch_id % 10 == 0:
            print("epoch:%d, batch:%d, cost:%.5f" %
                  (pass_id, batch_id, train_cost))
        iter = iter + BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0])
# 保存模型
model_save_dir = './model'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(model_save_dir, ['x'], [y_predict], exe)
plt.figure("Training Cost")
plt.title("Training Cost", fontsize=24)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.plot(iters, train_costs, color='red', label="Training Cost")
plt.grid()
plt.savefig("Training Cost.png")

# 4. 执行预测
infer_exe = fluid.Executor(place)  # 推理执行器
# 加载模型
inference_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(
    model_save_dir, infer_exe)

# 测试集读取器
infer_reader = paddle.dataset.uci_housing.test()
batch_infer_reader = paddle.batch(
    infer_reader, batch_size=200)
test_data = next(batch_infer_reader())

print(test_data)

text_x = np.array([x[0] for x in test_data]).astype(np.float32)
text_y = np.array([x[1] for x in test_data]).astype(np.float32)

params = {feed_target_names[0]: text_x}
results = infer_exe.run(inference_program,
                        feed=params,
                        fetch_list=fetch_targets)

infer_result = []  # 预测结果
ground_truth = []  # 真实值
for val in results[0]:
    infer_result.append(val)
for val in text_y:
    ground_truth.append(val)
print(infer_result, ground_truth)

plt.figure("Prediction")
plt.title("Prediction", fontsize=24)
plt.xlabel("House Price", fontsize=14)
plt.ylabel("Prediction", fontsize=14)
x = range(1, 30)
y = x
plt.plot(x, y, color='red', label="Ground Truth")
plt.scatter(ground_truth, infer_result, color='green', label="Prediction")
plt.grid()
plt.legend()
plt.savefig("Prediction.png")
plt.show()
