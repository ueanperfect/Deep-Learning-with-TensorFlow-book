import numpy as np

data = []
for i in range(100):
    x = np.random.uniform(-10,10)
    eps = np.random.normal(0,0.01)
    y = 2*x+0.089+eps
    data.append([x,y])
data = np.array(data)
print(data)
#计算误差
def mse(b,w,points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError +=(y-(w*x+b))**2
    return totalError/float(len(points))
#计算梯度
def step_graditent(b_current,w_current,points,lr):
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient +=(2/M)*((w_current*x+b_current)-y) # b_gradient += (2/N) * ((w_current * x + b_current) - y)
        w_gradient += (2/M)*x*((w_current*x+b_current)-y)
    new_b = b_current - (lr*b_gradient)
    new_w = w_current - (lr*w_gradient)
    return [new_b,new_w]
#梯度更新
def garatient_descent(points,starting_b,starting_w,lr,num_iterations):
    b=starting_b
    w=starting_w
    for step in range(num_iterations):
        b,w=step_graditent(b,w,np.array(points),lr)
        loss = mse(b,w,points)
        if step%50==0:
            print(f"iteration:{step},loss{loss},w:{w},b:{b}")
    return [b,w]
#最终实现：
def run():
    lr = 0.02
    ini_b=0
    ini_w=0
    times = 2000
    [b,w]=garatient_descent(data,ini_b,ini_w,lr,times)
    loss = mse(b,w,data)
    print(f'Final loss:{loss},w:{w},b:{b}')

if __name__ == '__main__':
    run()

