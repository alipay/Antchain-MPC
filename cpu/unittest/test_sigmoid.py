import unittest
import numpy as np

def sigmoid_sin_v2_numpy(x, M=256):
    N = 256
    X = np.linspace(-M, M, N, endpoint=False)  # -M to+M的256个值
    term = 6
    Sigmoid = 1 / (1 + np.exp(-X))
    Sm5 = Sigmoid - 0.5

    Sm5_odd = Sm5 * 1.0
    Sm5_odd[0] = 0
    F = np.fft.fft(Sm5_odd)
    # a1 = F[1].imag
    # a2 = F[2].imag
    # a3 = F[3].imag
    # a4 = F[4].imag
    # a5 = F[5].imag
    # print(a1, a2, a3, a4, a5)
    a = F[0:term].imag
    # a = tf.constant(a, dtype='float32', shape=[term]+[1]*len(x.shape))
    a = np.reshape(a, newshape=[term] + [1] * len(x.shape))
    # a = tf.constant(a, dtype='float32')

    print("a@sigmoid_sin_v2_numpy=", a)

    # a=[0, 79.01170242947703,  4.373381493344161, 21.67435635245209, 5.884090709554829,10.443694686327152]

    integers = np.reshape(range(term), newshape=[term] + [1] * len(x.shape))
    print("integers=", integers)
    x = np.expand_dims(x, axis=[0])

    # fake = -1.0 * (a[1] * sin2pi( (x - M) / (2 * M)) + a[2] * sin2pi(2 * (x - M) / (2 * M))
    #                + a[3] * sin2pi(3 * (x - M) / (2 * M)) + a[4] * sin2pi(4 * (x - M) / (2 * M))
    #                + a[5] * sin2pi(5 * (x - M) / (2 * M))) / 128
    y = integers * (x - M) / (2 * M)
    print("y@sigmoid_sin_v2:", y)
    fake = 0 - 1.0 * a * np.sin(2 * np.pi * y) / (N / 2)
    print("fake=", fake)
    # fake = fake.reduce_sum(axis=[0])
    fake = np.sum(fake, axis=0)
    print("1. fake=", fake)
    # fake = 0.5/(1.093936768972195-0.5) * fake + 0.5
    fake = fake + 0.5  # + 0.001*x
    print("2. fake=", fake)
    return fake

def sigmoid_poly(x):
    """A Chebyshev polynomial approximation of the sigmoid function."""

    w0 = 0.5
    w1 = 0.2159198015
    w3 = -0.0082176259
    w5 = 0.0001825597
    w7 = -0.0000018848
    w9 = 0.0000000072

    x1 = x
    x2 = (x1 * x)
    x3 = (x2 * x)
    x5 = (x2 * x3)
    x7 = (x2 * x5)
    x9 = (x2 * x7)

    y1 = w1 * x1
    y3 = w3 * x3
    y5 = w5 * x5
    y7 = w7 * x7
    y9 = w9 * x9

    z = y9 + y7 + y5 + y3 + y1 + w0

    return z


def sigmoid_sin_v3_numpy(x, M):
    N = 256
    X = np.linspace(-M, M, N, endpoint=False)  # -M to+M的256个值
    term = 6
    Sigmoid = 1 / (1 + np.exp(-X))
    Sm5 = Sigmoid - 0.5

    Sm5_odd = Sm5 * 1.0
    Sm5_odd[0] = 0
    F = np.fft.fft(Sm5_odd)
    a = F[0:term].imag
    a = np.array([pow(-1, n) for n in range(term)])*a
    a = -1.0*a/(N / 2)
    a = np.reshape(a, newshape=[term] + [1] * len(x.shape))

    print("a@sigmoid_sin_v2_numpy=", a)

    # a=[0, 79.01170242947703,  4.373381493344161, 21.67435635245209, 5.884090709554829,10.443694686327152]

    integers = np.reshape(range(term), newshape=[term] + [1] * len(x.shape))
    print("integers=", integers)
    x = np.expand_dims(x, axis=[0])

    # fake = (a[1] * sin2pi( x / (2 * M)) + a[2] * sin2pi(2 * x / (2 * M))
    #                + a[3] * sin2pi(3 * x  / (2 * M)) + a[4] * sin2pi(4 * x -  / (2 * M))
    #                + a[5] * sin2pi(5 * (x - M) / (2 * M)))
    y = integers * x / (2 * M)
    print("y@sigmoid_sin_v2:", y)
    fake = a * np.sin(2 * np.pi * y)
    print("fake=", fake)
    # fake = fake.reduce_sum(axis=[0])
    fake = np.sum(fake, axis=0)
    print("1. fake=", fake)
    fake = fake + 0.5
    print("2. fake=", fake)
    return fake

def sigmoid_line(x):
    y = (x<-0.5) * 0 + (x>=-0.5) * (x<0.5)*(x+0.5) + (x>=0.5) * 1
    return y


class MyTestCase(unittest.TestCase):

    def test_sigmoid(self):
        M = 16
        x = np.linspace(-5, 5, 256, endpoint=True)
        trues = 1 / (1 + np.exp(-x))
        fake_sigmoid = sigmoid_sin_v3_numpy(x, M)
        fake_poly = sigmoid_poly(x)
        fake_line = sigmoid_line(np.array(x))
        import matplotlib.pyplot as plt

        plt.plot(x, trues, ls='-', label='original sigmoid')
        plt.plot(x, fake_sigmoid, ls='--', label='sigmoid fourier')
        plt.plot(x, fake_poly, ls='-.', label='sigmoid poly')
        #plt.plot(x, fake_line, ls='-.', label='sigmoid piecewise linear')

        plt.legend()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()




if __name__ == '__main__':
    unittest.main()
