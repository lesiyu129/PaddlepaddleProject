import numpy
import paddle
import paddle.reader
paddle.enable_static()


def reader_creator(path):
    def reader():
        with open(path, 'r') as f:
            for line in f:
                yield line.replace('\n', '')

    return reader


reader = reader_creator('test.txt')
shuffle_reader = paddle.reader.shuffle(reader, buf_size=10)
batch_reader = paddle.batch(shuffle_reader, batch_size=3)
for data in batch_reader():
    print(data, end="")
