#applying kernel x on the matrix y.

#import numpy as np
#from scipy import signal as sg
#x = [6,2]
#y = [1,2,5,4]
#with padding full,same and no padding(valid)
##z  = np.convolve(x,y,"same") #z = np.convolve(x,y,"full") 
#z  = np.convolve(x,y,"valid")
#print z
#I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230]]
    
#g= [[-1, 1]]

#print sg.convolve(I,g,"valid")

imput = tf.Variable(tf.random_normal([1,10,10,1]))
filter = tf.Variable(tf.random_normal([3,3,1,1]))
op = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding = "same")
op2 = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding = "valid")

ini = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(ini)
    print input.eval()
    res = sess.run(op)
    res2 = sess.riun(op2)

    print res,res2
