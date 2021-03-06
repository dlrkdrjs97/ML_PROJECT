import tensorflow as tf

class PREDICT():
    def __init__(self,IN, label):
        self.IN = IN
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
   

    def predict(self, IN):
        input_layer = IN
        '''
        Layer1 = tf.matmul(input_layer, self.W1)+self.b1
        act_L1 = tf.nn.relu(Layer1)
        act_L1_drop = tf.nn.dropout(act_L1, 0.25)
        Layer2 = tf.matmul(act_L1_drop, self.W2)+self.b2
        act_L2 = tf.nn.relu(Layer2)
        act_L2_drop = tf.nn.dropout(act_L2, 0.25)
        Layer_out = tf.matmul(act_L2_drop, self.W3)+self.b3
        out = tf.sigmoid(Layer_out)
        '''
        out = tf.sigmoid(tf.matmul(input_layer, self.W)+self.b)
        return self.sess.run(out)
        
   
        
    def train(self):
        input_layer = tf.placeholder(dtype = tf.float32, shape = [None, 5])
        target = tf.placeholder(dtype = tf.float32, shape = [None, 1])
        '''
        self.W1 = self.weight_variable([5, 12])
        self.b1 = self.bias_variable([12])
        self.W2 = self.weight_variable([12,4])
        self.b2 = self.bias_variable([4])
        self.W3 = self.weight_variable([4,1])
        self.b3 = self.bias_variable([1])
        Layer1 = tf.matmul(input_layer, self.W1)+self.b1
        act_L1 = tf.nn.relu(Layer1)
        act_L1_drop = tf.nn.dropout(act_L1, 0.15)
        Layer2 = tf.matmul(act_L1_drop, self.W2)+self.b2
        act_L2 = tf.nn.relu(Layer2)
        act_L2_drop = tf.nn.dropout(act_L2, 0.23)
        Layer_out = tf.matmul(act_L2_drop, self.W3)+self.b3
        out = tf.sigmoid(Layer_out)
        '''
        self.W = self.weight_variable([5,1])
        self.b = self.bias_variable([1])
        out = tf.sigmoid(tf.matmul(input_layer, self.W)+self.b)

        loss = tf.reduce_mean(tf.square(target-out))#-tf.reduce_mean(target*tf.log(out))
        optimizer = tf.train.AdamOptimizer(1e-1)
        train = optimizer.minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
 
        for epoch in range(100):
            _, loss_temp = self.sess.run([train, loss], feed_dict = {input_layer: self.IN, target: self.label})
            print("epoch : {} Loss : {}".format(epoch, loss_temp))