import numpy as np


def gen_sample_data(dim, num_samples):
    w = np.random.rand(dim + 1) - 0.5
    # w *= 10                         #w is a dim-d vector
    x_list = 100 * np.random.rand(dim + 1, num_samples)         #random.randint(0, 100) * random.random()
    y_list = np.dot(w, x_list) + np.random.rand(num_samples)
    return x_list, y_list, w


def eval_loss(w, x, y):
    return (np.linalg.norm(np.dot(w, x) - y)) ** 2 / len(x)


# w -= l_r / x_list.size() * np.linalg.norm(np.dot(w, x_list) - y_list) * x_list
def cal_step_gradient(batch_x, batch_y, w, l_r):
    avg_dw = np.linalg.norm((np.dot(w, batch_x) - batch_y) * batch_x) / len(batch_x)
    w -= l_r * avg_dw
    return w


def train(x, y, w_init, batch_size, l_r, max_iterate):
    w = w_init
    for i in range(max_iterate):
        batch_i = np.random.choice(len(x), batch_size)
        batch_x = np.array([x[:, j] for j in batch_i]).T
        batch_y = [y[j] for j in batch_i]
        w = cal_step_gradient(batch_x, batch_y, w, l_r)
        print('loss is {0}'.format(eval_loss(w, x, y)))

def run():
    x, y, w = gen_sample_data(5, 10000)
    lr = 0.00001
    max_iter = 1000
    batch_size = 5
    train(x, y, w, batch_size, lr, max_iter)


if __name__ == '__main__':	# 跑.py的时候，跑main下面的；被导入当模块时，main下面不跑，其他当函数调
    run()