def complexity_softmax(m, n, r):
    """
    Calculates the communication costs for softmax protocol.
    Returns offline, online, overall costs, and rounds of communication

    The offline communication complexity is 2mnr.
    The offline communication complexity is (6mn + 2n)r.
    The rounds of communication are 2r.

    Args:
        m (int): number of classes.
        n (int): bitlength of fixed-point numbers, usually n=64.
        r (int): rounds of iterations.
    """

    offline = 2 * m * n * r
    online = (6 * m * n + 2 * n) * r
    round = 2 * r
    return offline, online, offline + online, round


def complexity_sigmoid(m, n, K, f):
    """
    Calculates the communication costs for sigmoid protocol.
    Returns offline, online, overall costs, and rounds of communication

    The offline communication complexity is 2Kn.
    The offline communication complexity is 2(m+f).
    The rounds of communication constantly equals 1.

    Args:
        m (int): variable relevant to period.
        n (int): bitlength of fixed-point numbers, usually n=64.
        K (int): indicates the K+1 terms for approximation.
        f (int): fractional parts.
    """

    offline = 2 * K * n
    online = 2 * (m + f)
    round = 1
    return offline, online, offline + online, round


def get_complexity_softmax():
    """
    Calculates and prints the communication costs for softmax protocol.

    n is constantly set to 64.
    """

    print("Softmax communication:")
    print("m     n   r   offline  online    overall   round")
    n = 64
    for m in [10, 100, 1000]:
        for r in [8, 16, 32, 64]:
            result = complexity_softmax(m, n, r)
            print("%-5d %-3d %-3d %-8d %-9d %-9d %-4d" % (m, n, r, result[0], result[1], result[2], result[3]))
    print()


def get_complexity_sigmoid():
    """
    Calculates and prints the communication costs for sigmoid protocol.

    n is constantly set to 64.
    f is 14 in our work.
    """

    print("Sigmoid communication:")
    print("m  n   K  f   offline  online    overall   round")
    n, f = 64, 14
    for m in [4, 5]:
        for K in [5, 8]:
            result = complexity_sigmoid(m, n, K, f)
            print("%-2d %-3d %-2d %-3d %-8d %-9d %-9d %-4d" % (m, n, K, f, result[0], result[1], result[2], result[3]))
    print()


if __name__ == "__main__":
    get_complexity_softmax()
    get_complexity_sigmoid()
