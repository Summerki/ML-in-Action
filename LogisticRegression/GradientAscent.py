# 梯度上升算法实现


# 梯度上升算法
# 求函数f(x) = -x^2 + 4x的极大值
def Gradient_Ascent_test():
    x_old = -1  # 初始值
    x_new = 0  # 梯度上升算法初始值
    alpha = 0.01  # 步长
    presision = 0.00000001  # 精度，也就是更新的阈值
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)



def f_prime(x_old):
    return -2 * x_old + 4





if __name__ == '__main__':
    Gradient_Ascent_test()