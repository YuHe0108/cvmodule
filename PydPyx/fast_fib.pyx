import functools

def fib1(n):
    if n in [1, 2]:
        return 1
    return fib1(n - 1) + fib1(n - 2)

def fib2():
    a = 0
    b = 1

    def calc():
        nonlocal a, b
        a, b = b, a + b
        return a

    return calc

@functools.lru_cache()  # 在fibonacci函数上加上装饰器函数functools.lru_cache()
def fib3(n):
    if n in [1, 2]:
        return 1
    return fib3(n - 1) + fib3(n - 2)
