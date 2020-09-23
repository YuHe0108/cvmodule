"""绘制动态图"""
import matplotlib.pyplot as plt
import time


def plot_by_epoch(values, epochs, y_label, x_label='epoch'):
    """绘制value随着epoch动态变换的情况"""
    plt.ion()
    for i in range(1, len(epochs)):
        ix = values[:i]
        iy = epochs[:i]
        plt.plot(ix, iy)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pause(0.001)

    plt.ioff()
    plt.show()
    return

if __name__ == '__main__':
    epochs = list(range(1, 100))  # epoch array
    values = [10 / (i ** 2) for i in epochs]  # loss values array
    plot_by_epoch(values, epochs, 'loss')
