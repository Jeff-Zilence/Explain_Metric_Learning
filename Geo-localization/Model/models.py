import tensorflow as tf

## The code is modified from CVM-Net, better model and cleaner code are coming soon!

def Siamese_network(x_sat, x_grd):

    with tf.variable_scope('Sat') as scope:
        vgg_sat = VGG16()
        sat_local = vgg_sat.build(x_sat)
    with tf.variable_scope('Grd') as scope:
        vgg_grd = VGG16()
        grd_local = vgg_grd.build(x_grd)

    # Global Average Pooling
    with tf.variable_scope('Gap_local') as scope:
        gap_sat = tf.reduce_mean(sat_local,axis=[1,2])
        gap_grd = tf.reduce_mean(grd_local,axis=[1,2])

        fc_sat = fc_layer(gap_sat, 512, 4096, True, 'fc1', activation_fn=None)
        sat_global = tf.nn.l2_normalize(fc_sat, dim=1)

        scope.reuse_variables()

        fc_grd = fc_layer(gap_grd, 512, 4096, True, 'fc1', activation_fn=None)
        grd_global = tf.nn.l2_normalize(fc_grd, dim=1)
    return sat_global, grd_global, sat_local, grd_local, fc_sat, fc_grd

def fc_layer(x, input_dim, output_dim,
                 trainable, name='fc_layer', activation_fn=tf.nn.relu,reuse = tf.AUTO_REUSE):
    with tf.variable_scope(name,reuse = reuse):
        weight = tf.get_variable(name='weights', shape=[input_dim, output_dim],
                                     trainable=trainable,
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

        if activation_fn is not None:
            out = tf.nn.xw_plus_b(x, weight, bias)
            out = activation_fn(out)
        else:
            out = tf.nn.xw_plus_b(x, weight, bias)
    return out

class VGG16:

    ############################ kernels #############################
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                            padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    ############################ layers ###############################
    def conv_layer(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            if activated:
                out = activation_function(self.conv2d(x, weight) + bias)
            else:
                out = self.conv2d(x, weight) + bias

            return out

    def maxpool_layer(self, x, name):
        with tf.name_scope(name):
            maxpool = self.max_pool_2x2(x)
            return maxpool


    # the convolutional part of VGG16-D
    def build(self, x):

        # layer 1: conv3-64
        layer1_output = self.conv_layer(x, 3, 3, 64, True, True, 'conv1_1')
        # layer 2: conv3-64
        layer2_output = self.conv_layer(layer1_output, 3, 64, 64, True, True, 'conv1_2')
        # layer3: max pooling
        layer3_output = self.maxpool_layer(layer2_output, 'layer3_maxpool2x2')

        # layer 4: conv3-128
        layer4_output = self.conv_layer(layer3_output, 3, 64, 128, True, True, 'conv2_1')
        # layer 5: conv3-128
        layer5_output = self.conv_layer(layer4_output, 3, 128, 128, True, True, 'conv2_2')
        # layer 6: max pooling
        layer6_output = self.maxpool_layer(layer5_output, 'layer6_maxpool2x2')

        # layer 7: conv3-256
        layer7_output = self.conv_layer(layer6_output, 3, 128, 256, True, True, 'conv3_1')
        # layer 8: conv3-256
        layer8_output = self.conv_layer(layer7_output, 3, 256, 256, True, True, 'conv3_2')
        # layer 9: conv3-256
        layer9_output = self.conv_layer(layer8_output, 3, 256, 256, True, True, 'conv3_3')
        # layer 10: max pooling
        layer10_output = self.maxpool_layer(layer9_output, 'layer10_maxpool2x2')

        # layer 11: conv3-512
        layer11_output = self.conv_layer(layer10_output, 3, 256, 512, True, True, 'conv4_1')
        # layer 12: conv3-512
        layer12_output = self.conv_layer(layer11_output, 3, 512, 512, True, True, 'conv4_2')
        # layer 13: conv3-512
        layer13_output = self.conv_layer(layer12_output, 3, 512, 512, True, True, 'conv4_3')
        # layer 14: max pooling
        layer14_output = self.maxpool_layer(layer13_output, 'layer14_maxpool2x2')

        # layer 15: conv3-512
        layer15_output = self.conv_layer(layer14_output, 3, 512, 512, True, True, 'conv5_1')
        # layer 16: conv3-512
        layer16_output = self.conv_layer(layer15_output, 3, 512, 512, True, True, 'conv5_2')
        # layer 17: conv3-512
        layer17_output = self.conv_layer(layer16_output, 3, 512, 512, True, True, 'conv5_3')

        return layer17_output