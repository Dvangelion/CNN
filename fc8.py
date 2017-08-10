import numpy as np
import tensorflow as tf
import random



# maxpool = np.load('maxpool.npy')
# maxpoolr = np.load('maxpool_r.npy')
# maxpoola = np.load('maxpool_a.npy')
# maxpoolb = np.load('maxpool_b.npy')
# maxpoola = maxpoola.reshape(-1,6*6*256)
# maxpoolr = maxpoolr.reshape(-1,6*6*256)
#
# train_data = np.append(maxpool[:6900],maxpoolr,axis=0)
# train_data = np.append(train_data,maxpoola,axis=0)
# train_data = np.append(train_data,maxpoolb,axis=0)

train_data = np.load('fc7_total.npy')

train_label_raw = np.load('train_label.npy')
train_label = np.append(train_label_raw[:6900],train_label_raw[:1000],axis=0)
train_label = np.append(train_label,train_label[:3000],axis=0)
train_label = np.append(train_label,train_label[:4000],axis=0)

r = random.random()
random.shuffle(train_data, lambda : r)  # lambda : r is an unary function which returns r
random.shuffle(train_label, lambda : r)  # using the same function as used in prev line so that shuffling order is same


n_input = 9216
n_hidden1 = 4096
n_hidden2 = 2048
dropout = .8

num_examples = 14900
learning_rate = .001
traning_epochs = 15
batch_size = 200
display_step = 1
n_classes = 8
keep_prob = tf.placeholder(tf.float32)


x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,8])



# fc8W = tf.Variable(tf.random_normal([4096,8]))
# fc8b = tf.Variable(tf.random_normal([8]))

def perceptron(x,weights,biases):

    layer1 = tf.add(tf.matmul(x,weights['h1']) , biases['h1'])
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.dropout(layer1,dropout)

    layer2 = tf.add(tf.matmul(layer1,weights['h2']) , biases['h2'])
    layer2 = tf.nn.relu(layer2)

    layer2 = tf.nn.dropout(layer2,dropout)

    out = tf.add(tf.matmul(layer2,weights['o']) , biases['o'])
    #out = tf.nn.softmax(out)

    return  out

weights = {
    'h1': tf.Variable(tf.random_normal([n_input,n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
    'o' : tf.Variable(tf.random_normal([n_hidden2,n_classes]))
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden2])),
    'o' : tf.Variable(tf.random_normal([n_classes]))
}

pred = perceptron(x,weights,biases)

#pred = tf.nn.softmax(tf.matmul(x, fc8W) + fc8b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    train_cost = []
    val_cost = []
    train_acc = []
    val_acc = []



    for epoch in range(traning_epochs):
        avg_cost = 0
        total_batch = int(num_examples/batch_size)
        r1 = random.random()
        #random.shuffle(train_data, lambda : r1)  # lambda : r is an unary function which returns r
        #random.shuffle(train_label, lambda : r1)  # using the same function as used in prev line so that shuffling order is same


        minibatch_loss = []
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

            train_c,train_a = sess.run([cost,accuracy],feed_dict={x:maxpool[:100],y:train_label_raw[:100]})
            val_c,val_a = sess.run([cost,accuracy],feed_dict={x:maxpool[-100:],y:train_label_raw[-100:]})

            print 'training acc: ' ,train_a,'training cost: ',avg_cost
            print 'val acc: ',val_a, 'val cost: ',val_c

            train_acc.append(train_a)
            train_cost.append(train_c)
            val_cost.append(val_c)
            val_acc.append(val_a)

    np.save('score.npy',(train_acc,train_cost,val_acc,val_cost))

    print "Training Metric saved. "


    print "Optimization done. "

    #np.save('minibatchloss.npy',minibatch_loss)

    fc6W = sess.run(weights['h1'])
    fc6b = sess.run(biases['h1'])
    np.save('fc6W.npy',fc6W)
    np.save('fc6b.npy',fc6b)

    fc7W = sess.run(weights['h2'])
    fc7b = sess.run(biases['h2'])

    np.save('fc7W.npy',fc7W)
    np.save('fc7b.npy',fc7b)

    fc8W = sess.run(weights['o'])
    fc8b = sess.run(biases['o'])
    np.save('fc8W.npy',fc8W)
    np.save('fc8b.npy',fc8b)

    #save_path = saver.save(sess,"/tmp/perceptron.ckpt")
    print "Model saved. "

