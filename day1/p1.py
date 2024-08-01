import paddle.fluid as fluid
import numpy as np
import paddle
# 切换到静态图模式
paddle.enable_static()

x = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')
y = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')

x_add_y = fluid.layers.elementwise_add(x, y)
x_mul_y = fluid.layers.elementwise_mul(x, y)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])

params = {'x': a, 'y': b}
outs = exe.run(fluid.default_main_program(),
               feed=params,
               fetch_list=[x_add_y, x_mul_y])

print(outs)
