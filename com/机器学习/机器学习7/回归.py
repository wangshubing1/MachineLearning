import math
import matplotlib.pyplot as plt


def f(w, x):
    N = len(w)
    i = 0
    y = 0
    while i < N - 1:
        y += w[i] * x[i]
        i += 1
    y += w[N - 1]  # 常数项
    return y


def gradient(data, w, j):
    M = len(data)  # 样本数
    N = len(data[0])
    i = 0
    g = 0  # 当前维度的梯度
    while i < M:
        y = f(w, data[i])
        if (j != N - 1):
            g += (data[i][N - 1] - y) * data[i][j]
        else:
            g += data[i][N - 1] - y
        i += 1
    return g / M


def gradientStochastic(data, w, j):
    N = len(data)  # 维度
    y = data[N - 1] - f(w, data)
    if (j != N - 1):
        return y * data[j]
    return y  # 常数项


def isSame(a, b):
    n = len(a)
    i = 0
    while i < n:
        if abs(a[i] - b[i]) > 0.01:
            return False
        i += 1
    return True


def fw(w, data):
    M = len(data)  # 样本数
    N = len(data[0])
    i = 0
    s = 0
    while i < M:
        y = data[i][N - 1] - f(w, data[i])
        s += y ** 2
        i += 1
    return s / 2


def fwStochastic(w, data):
    y = data[len(data) - 1] - f(w, data)
    y **= 2
    return y / 2


def numberProduct(n, vec, w):
    N = len(vec)
    i = 0
    while i < N:
        w[i] += vec[i] * n
        i += 1


def assign(a):
    L = []
    for x in a:
        L.append(x)
    return L


# a = b
def assign2(a, b):
    i = 0
    while i < len(a):
        a[i] = b[i]
        i += 1


def dotProduct(a, b):
    N = len(a)
    i = 0
    dp = 0
    while i < N:
        dp += a[i] * b[i]
        i += 1
    return dp

#学习率
# w当前值；g当前梯度方向；a当前学习率；data数据
def calcAlpha(w, g, a, data):
    c1 = 0.3
    now = fw(w, data)
    wNext = assign(w)
    numberProduct(a, g, wNext)
    next = fw(wNext, data)
    # 寻找足够大的a，使得h(a)>0
    count = 30
    while next < now:
        a *= 2
        wNext = assign(w)
        numberProduct(a, g, wNext)
        next = fw(wNext, data)
        count -= 1
        if count == 0:
            break

            # 寻找合适的学习率a
    count = 50
    while next > now - c1 * a * dotProduct(g, g):
        a /= 2
        wNext = assign(w)
        numberProduct(a, g, wNext)
        next = fw(wNext, data)

        count -= 1
        if count == 0:
            break
    return a

#SGD
def calcAlphaStochastic(w, g, a, data):
    c1 = 0.01  # 因为是每个样本都下降，所以参数运行度大些，即：激进一些
    now = fwStochastic(w, data)
    wNext = assign(w)
    numberProduct(a, g, wNext)
    next = fwStochastic(wNext, data)
    # 寻找足够大的a，使得h(a)>0
    count = 30
    while next < now:
        if a < 1e-10:
            a = 0.01
        else:
            a *= 2
        wNext = assign(w)
        numberProduct(a, g, wNext)
        next = fwStochastic(wNext, data)
        count -= 1
        if count == 0:
            break

            # 寻找合适的学习率a
    count = 50
    while next > now - c1 * a * dotProduct(g, g):
        a /= 2
        wNext = assign(w)
        numberProduct(a, g, wNext)
        next = fwStochastic(wNext, data)

        count -= 1
        if count == 0:
            break
    return a


def normalize(g):
    s = 0
    for x in g:
        s += x * x
    s = math.sqrt(s)
    i = 0
    N = len(g)
    while i < N:
        g[i] /= s
        i += 1
#回归
# w当前值；g当前梯度方向；a当前学习率；data数据
def calcCoefficient(data, listA, listW, listLostFunction):
    M = len(data)  # 样本数目
    N = len(data[0])  # 维度
    w = [0 for i in range(N)]
    wNew = [0 for i in range(N)]
    g = [0 for i in range(N)]

    times = 0
    alpha = 100.0  # 学习率随意初始化
    same = False
    while times < 10000:
        i = 0
        while i < M:
            j = 0
            while j < N:
                g[j] = gradientStochastic(data[i], w, j)
                j += 1
            normalize(g)  # 正则化梯度
            alpha = calcAlphaStochastic(w, g, alpha, data[i])
            # alpha = 0.01
            numberProduct(alpha, g, wNew)

            print
            "times,i, alpha,fw,w,g:\t", times, i, alpha, fw(w, data), w, g
            if isSame(w, wNew):
                if times > 5:  # 防止训练次数过少
                    same = True
                    break
            assign2(w, wNew)  # 更新权值

            listA.append(alpha)
            listW.append(assign(w))
            listLostFunction.append(fw(w, data))

            i += 1
        if same:
            break
        times += 1
    return w


if __name__ == "__main__":
    fileData = open("d8.txt")
    data = []
    for line in fileData:
        d = map(float, line.split('\t'))
        data.append(d)
    fileData.close()

    listA = []  # 每一步的学习率
    listW = []  # 每一步的权值
    listLostFunction = []  # 每一步的损失函数值
    w = calcCoefficient(data, listA, listW, listLostFunction)

    # 绘制学习率
    plt.plot(listA, 'r-', linewidth=2)
    plt.plot(listA, 'go')
    plt.xlabel('Times')
    plt.ylabel('Ratio/Step')
    plt.show()

    # 绘制损失
    listLostFunction = listLostFunction[0:100]
    listLostFunction[0] /= 2
    plt.plot(listLostFunction, 'r-', linewidth=2)
    plt.plot(listLostFunction, 'gv', alpha=0.75)
    plt.xlabel('Times')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.show()

    # 绘制权值
    X = []
    Y = []
    for d in data:
        X.append(d[0])
        Y.append(d[1])
    plt.plot(X, Y, 'cp', label=u'Original Data', alpha=0.75)
    x = [min(X), max(X)]
    y = [w[0] * x[0] + w[1], w[0] * x[1] + w[1]]
    plt.plot(x, y, 'r-', linewidth=3, label='Regression Curve')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()