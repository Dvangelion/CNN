import tensorflow as tf
import numpy as np
import random

sess = tf.InteractiveSession()

train_data_raw = np.load("train_data.npy")
aug_data = np.load('aug_data.npy')
aug_data_1 = np.load('aug_data_1.npy')
rotate = np.load('rotate.npy')
train_label = np.load("train_label.npy")

train_data = np.append(train_data_raw[:6900],rotate,axis=0)
train_data = np.append(train_data,aug_data,axis=0)
train_data = np.append(train_data,aug_data_1,axis=0)

train_label_raw = np.load('train_label.npy')
train_label = np.append(train_label_raw[:6900],train_label[:1000],axis=0)
train_label = np.append(train_label,train_label[:3000],axis=0)
train_label = np.append(train_label,train_label[:4000],axis=0)

r = random.random()
random.shuffle(train_data, lambda : r)  # lambda : r is an unary function which returns r
random.shuffle(train_label, lambda : r)  # using the same function as used in prev line so that shuffling order is same


num_examples = 14900
training_epochs = 1
batch_size = 200
learning_rate = 0.001
n_input = 784
n_classes = 8
display_step = 100
dropout = 0.5

x = tf.placeholder(tf.float32,[None,227,227,3])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)



def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    x = tf.nn.relu(x)
    return x

def maxpool2d(x,k=2):
    x = tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
    return x

def conv_net(x,weights,biases,dropout):

    #x = tf.reshape(x,shape=[-1,28,28,1])
    x = tf.reshape(x,shape=[-1,227,227,3])

    conv1 = conv2d(x,weights['wc1'],biases['wc1'])
    conv1 = maxpool2d(conv1,k=4)

    conv2 = conv2d(conv1,weights['wc2'],biases['wc2'])
    conv2 = maxpool2d(conv2,k=4)


    fc1 = tf.reshape(conv2,[-1,weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['fc1']) , biases['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,dropout)

    out = tf.add(tf.matmul(fc1,weights['out']) , biases['out'])

    return out


weights = {
    'wc1':tf.Variable(tf.random_normal([3,3,1,32])),
    'wc2':tf.Variable(tf.random_normal([3,3,32,64])),
    'fc1':tf.Variable(tf.random_normal([8*8*64,2048])),
    'out':tf.Variable(tf.random_normal([2048,n_classes]))
}

biases = {
    'wc1':tf.Variable(tf.random_normal([32])),
    'wc2':tf.Variable(tf.random_normal([64])),
    'fc1':tf.Variable(tf.random_normal([2048])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

pred = conv_net(x,weights,biases,keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver()



with tf.Session() as sess:
    sess.run(init)


    for epoch in range(training_epochs):
        print 'training...'
        avg_cost = 0
        total_batch = int(num_examples/batch_size)

        train_cost = []
        val_cost = []
        train_acc = []
        val_acc = []

        #print "total batch:", total_batch
        for i in range(total_batch):

            batch_x = train_data[i * batch_size : (i+1)*batch_size]
            #print batch_x.shape


            batch_y = train_label[i * batch_size : (i+1)*batch_size]
            #print batch_y.shape

            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})





            avg_cost += c/(2*total_batch)

            # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # cost_list.append(avg_cost)
            # accuracy_list.append(accuracy.eval({x: mnist.test.images[:100], y: mnist.test.labels[:100]}))

        if epoch % display_step == 0:
            print "Epoch", "%04d" % (epoch+1),"cost=","{:.9f}".format(avg_cost)

            train_a = sess.run([accuracy],feed_dict={x:train_data_raw[:100],y:train_label_raw[:100]})
            val_c,val_a = sess.run([cost,accuracy],feed_dict={x:train_data_raw[-100:],y:train_label_raw[-100:]})

            print 'training acc: ' ,train_a,'training cost: ',avg_cost
            print 'val acc: ',val_a, 'val cost: ',val_cost

            train_acc.append(train_a)
            train_cost.append(avg_cost)
            val_cost.append(val_c)
            val_acc.append(val_a)

            np.save('score.npy',(train_acc,train_cost,val_acc,val_cost))

            np.save('model.npy',(weights,biases))









