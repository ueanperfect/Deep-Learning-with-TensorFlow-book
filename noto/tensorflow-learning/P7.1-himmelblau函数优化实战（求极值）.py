#himmelblau函数优化实战
import tensorflow as tf
def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

x = tf.Variable([4.,0.])

for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    grads = tape.gradient(y,[x])[0]#这边的x最好是Var变量
    x = x - 0.01*grads
    if step%10 == 0:
        print(x,y)

