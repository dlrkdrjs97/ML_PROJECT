import tensorflow as tf

class PREDICT():
    def __init__(self,IN1,IN2,IN3,IN4,IN5,label):
        self.IN1 = IN1
        self.IN2 = IN2
        self.IN3 = IN3
        self.IN4 = IN4
        self.IN5 = IN5
        self.label = label


    def training_process(self):
        self.train()
   
    
    # initialize variables
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
   

    def predict(self, IN1,IN2,IN3,IN4,IN5):
        input1 = tf.placeholder(tf.float32,[None, ])
        input2 = tf.placeholder(tf.float32,[None, ])
        input3 = tf.placeholder(tf.float32,[None, ])
        input4 = tf.placeholder(tf.float32,[None, ])
        input5 = tf.placeholder(tf.float32,[None, ])
        train_data =tf.stack([input1,input2,input3, input4,input5],axis=-1,name = 'stack')
        
        Layer1 = tf.matmul(train_data, self.W1)+self.b1
        act_L1 = tf.nn.relu(Layer1)
        act_L1_drop = tf.nn.dropout(act_L1, 0.25)
        Layer2 = tf.matmul(act_L1_drop, self.W2)+self.b2
        act_L2 = tf.nn.relu(Layer2)
        act_L2_drop = tf.nn.dropout(act_L2, 0.25)
        Layer_out = tf.matmul(act_L2_drop, self.W3)+self.b3
        out = tf.sigmoid(Layer_out)
        return self.sess.run(out)
        
   
        
    def train(self):
        input1 = tf.placeholder(tf.float32,[None, 1])
        input2 = tf.placeholder(tf.float32,[None, 1])
        input3 = tf.placeholder(tf.float32,[None, 1])
        input4 = tf.placeholder(tf.float32,[None, 1])
        input5 = tf.placeholder(tf.float32,[None, 1])
        target = tf.placeholder(tf.float32,[None, 1])
        train_data =tf.stack([input1,input2,input3, input4,input5],axis=-1,name = 'stack') 
        self.W1 = self.weight_variable([5, 12])
        self.b1 = self.bias_variable([12])
        self.W2 = self.weight_variable([12,4])
        self.b2 = self.bias_variable([4])
        self.W3 = self.weight_variable([4,1])
        self.b3 = self.bias_variable([1])
        Layer1 = tf.matmul(train_data, self.W1)+self.b1
        act_L1 = tf.nn.relu(Layer1)
        act_L1_drop = tf.nn.dropout(act_L1, 0.25)
        Layer2 = tf.matmul(act_L1_drop, self.W2)+self.b2
        act_L2 = tf.nn.relu(Layer2)
        act_L2_drop = tf.nn.dropout(act_L2, 0.25)
        Layer_out = tf.matmul(act_L2_drop, self.W3)+self.b3
        out = tf.sigmoid(Layer_out)

        loss = -tf.reduce_sum(target*tf.log(out))
        optimizer = tf.train.AdamOptimizer(0.001)
        train = optimizer.minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
 
        for epoch in range(100):
            _, loss_temp = self.sess.run([train, loss], feed_dict = {input1: self.IN1,input2: self.IN2, input3:self.IN3, input4:self.IN4, input5:self.IN5, target:self.label})
            print("epoch : {} Loss : ".format(epoch, loss_temp))