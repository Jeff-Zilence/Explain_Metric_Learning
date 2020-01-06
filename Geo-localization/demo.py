import os,sys
from Model import Siamese_network
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

## Better model and cleaner code are coming soon!

class Explanation_generator:

    def __init__(self, mode='CVUSA'):
        self.mode = mode
        self.size_sat = [512, 512]
        self.size_grd = [224, 1232]
        self.load_model()
        self.Decomposition = None

    # read image, subtract bias, convert to rgb for imshow
    def read(self, path, size):
        image = cv2.imread(path).astype(np.float)
        image = cv2.resize(image,(size[1],size[0]))
        image_show = image[:,:,::-1]/255.
        image = image - np.array([103.939, 116.779, 123.6])
        return np.expand_dims(image,axis=0), image_show

    # if you wanna load data from CVUSA dataset, you may download the dataset and write a simple dataloader.
    def get_input_from_path(self, path_1, path_2):
        '''
            load two images from paths
            sat denotes satallite (Aerial view) and grd denotes ground (Street view)
        '''
        inputs_sat, image_sat = self.read(path_1, size=self.size_sat)
        inputs_grd, image_grd = self.read(path_2, size=self.size_grd)

        return inputs_sat, image_sat, inputs_grd, image_grd

    def load_model(self, path = 'Geo-localization/Model/model.ckpt'):
        '''
            Load the trained model, you may change the path of model here.
            Get the cosine similarity and parameters of fc layer
        '''
        self.sat_x = tf.placeholder(tf.float32, [None, self.size_sat[0], self.size_sat[1], 3], name='sat_x')
        self.grd_x = tf.placeholder(tf.float32, [None, self.size_grd[0], self.size_grd[1], 3], name='grd_x')
        self.keep_prob = tf.placeholder(tf.float32)

        self.sat_global, self.grd_global, self.sat_local, self.grd_local, self.fc_sat, self.fc_grd = Siamese_network(self.sat_x, self.grd_x)
        self.product = tf.reduce_sum(self.sat_global*self.grd_global,axis=1)
        self.product_ori = tf.reduce_sum(self.fc_sat*self.fc_grd,axis=1)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

        print('load model...')
        saver.restore(self.sess, path)
        print("Model loaded from: %s" % path)

        self.fc_var = [v for v in tf.global_variables() if 'fc1' in v.name]
        self.fc_weight, self.fc_bias = self.sess.run(self.fc_var)

    def imshow_convert(self, raw):
        '''
            convert the heatmap for imshow
        '''
        heatmap = np.array(cv2.applyColorMap(np.uint8(255*(1.-raw)), cv2.COLORMAP_JET))
        return heatmap


    def GradCAM(self, sess, cost , target, feed_dict, size):
        gradient = tf.gradients(cost, target)[0]
        conv_output, conv_first_grad = sess.run([target, gradient], feed_dict=feed_dict)

        # compute the average value
        weights = np.mean(conv_first_grad[0], axis = (0, 1)) 
        grad_CAM_map = np.sum(weights*conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0   
        cam = cv2.resize(cam, (size[1],size[0]))
        return cam

    def RGradCAM(self, sess, cost , target, feed_dict, size):
        # rectified Grad-CAM, a variant
        gradient = tf.gradients(cost, target)[0]
        conv_output, conv_first_grad = sess.run([target, gradient], feed_dict=feed_dict)

        # remove the heuristic GAP step
        weights = conv_first_grad[0]
        grad_CAM_map = np.sum(weights*conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = cv2.resize(cam, (size[1], size[0]))
        return cam

    def Overall_map(self, map_1, map_2, size = (256, 128), mode = 'GAP'):
        '''
            Only for GAP architecture, you may check the code of other applications 
            for the implementation of GMP and flattened feature.
        '''
        if mode == 'GAP':

            map_1_reshape = np.reshape(map_1, [-1, map_1.shape[-1]])
            map_2_reshape = np.reshape(map_2, [-1, map_2.shape[-1]])
            # initialize the equivalent weight and bias
            weight_1 = self.fc_weight
            bias_1 = self.fc_bias
            weight_2 = self.fc_weight
            bias_2 = self.fc_bias

            # compute the transformed feature
            map_1_embed = np.matmul(map_1_reshape, weight_1)
            map_2_embed = np.matmul(map_2_reshape, weight_2) 
            # reshape back
            map_1_embed = np.reshape(map_1_embed, [map_1.shape[1], map_1.shape[2], -1])
            map_2_embed = np.reshape(map_2_embed, [map_2.shape[1], map_2.shape[2], -1])

            Decomposition = np.zeros([map_1.shape[1],map_1.shape[2],map_2.shape[1],map_2.shape[2]])
            for i in range(map_1.shape[1]):
                for j in range(map_1.shape[2]):
                    for x in range(map_2.shape[1]):
                        for y in range(map_2.shape[2]):
                            Decomposition[i,j,x,y] = np.sum(map_1_embed[i,j]*map_2_embed[x,y])
            Decomposition = Decomposition / np.max(Decomposition)
            Decomposition = np.maximum(Decomposition, 0)
            return Decomposition

    def Point_Specific(self, decom, point = [0,0], stream = 1, size=(256, 128)):
        '''
            Generate the point-specific activation map
        '''
        if stream == 2:
            decom_padding = np.pad(np.transpose(decom,(2,3,0,1)), ((1,1),(1,1),(0,0),(0,0)), mode='edge')
        else:
            decom_padding = np.pad(decom, ((1,1),(1,1),(0,0),(0,0)), mode='edge')
        # compute the transformed coordinates
        x = (point[0] + 0.5) / size[0] * (decom_padding.shape[0]-2)
        y = (point[1] + 0.5) / size[1] * (decom_padding.shape[1]-2)
        x = x + 0.5
        y = y + 0.5
        x_min = int(np.floor(x))
        y_min = int(np.floor(y))
        x_max = x_min + 1
        y_max = y_min + 1
        dx = x - x_min
        dy = y - y_min
        interplolation = decom_padding[x_min, y_min]*(1-dx)*(1-dy) + \
                         decom_padding[x_max, y_min]*dx*(1-dy) + \
                         decom_padding[x_min, y_max]*(1-dx)*dy + \
                         decom_padding[x_max, y_max]*dx*dy
        return np.maximum(interplolation,0)

    def demo(self, path_1='Geo-localization/Images/0013073_sat.jpg', \
                   path_2='Geo-localization/Images/0013073_grd.jpg'):
        '''
            generate activation map with different methods.
        '''
        inputs_sat, image_1, inputs_grd, image_2 = self.get_input_from_path(path_1=path_1, path_2=path_2)
        feed_dict = {self.sat_x: inputs_sat, self.grd_x: inputs_grd}

        #--------------------------------------------------------------------------------
        '''
            Since the model is based on global average pooling and trained without L2 normalizationa,
            the result of activation decomposition with bias ("Decomposition+Bias") is 
            equivalent with Grad-CAM and RGradCAM without normalization
        '''
        # grad-cam with normalization
        gradcam_1 = self.GradCAM(sess=self.sess, cost=self.product, target=self.sat_local, feed_dict=feed_dict, size=self.size_sat)
        gradcam_2 = self.GradCAM(sess=self.sess, cost=self.product, target=self.grd_local, feed_dict=feed_dict, size=self.size_grd)

        image_overlay_1 = image_1 * 0.7 + self.imshow_convert(gradcam_1) / 255.0 * 0.3
        image_overlay_2 = image_2 * 0.7 + self.imshow_convert(gradcam_2) / 255.0 * 0.3

        plt.figure()
        plt.suptitle('Grad-CAM')
        plt.subplot(2,2,1)
        plt.imshow(self.imshow_convert(gradcam_1))
        plt.subplot(2,2,2)
        plt.imshow(self.imshow_convert(gradcam_2))
        plt.subplot(2,2,3)
        plt.imshow(image_overlay_1)
        plt.subplot(2,2,4)
        plt.imshow(image_overlay_2)

        # grad-cam without normalization
        gradcam_1 = self.GradCAM(sess=self.sess, cost=self.product_ori, target=self.sat_local, feed_dict=feed_dict, size=self.size_sat)
        gradcam_2 = self.GradCAM(sess=self.sess, cost=self.product_ori, target=self.grd_local, feed_dict=feed_dict, size=self.size_grd)

        image_overlay_1 = image_1 * 0.7 + self.imshow_convert(gradcam_1) / 255.0 * 0.3
        image_overlay_2 = image_2 * 0.7 + self.imshow_convert(gradcam_2) / 255.0 * 0.3

        plt.figure()
        plt.suptitle('Grad-CAM without normalization (Decomposition+Bias)')
        plt.subplot(2,2,1)
        plt.imshow(self.imshow_convert(gradcam_1))
        plt.subplot(2,2,2)
        plt.imshow(self.imshow_convert(gradcam_2))
        plt.subplot(2,2,3)
        plt.imshow(image_overlay_1)
        plt.subplot(2,2,4)
        plt.imshow(image_overlay_2)

        #--------------------------------------------------------------------------------
        '''
            Generate overall activation map using activation decomposition ("Decomposition"),
        '''

        # compute the overall activation map with decomposition (no bias term)
        map_1, map_2, similarity = self.sess.run([self.sat_local, self.grd_local, self.product], feed_dict=feed_dict)
        print(similarity)
        self.Decomposition = self.Overall_map(map_1 = map_1, map_2 = map_2, mode = 'GAP')

        decom_1 = cv2.resize(np.sum(self.Decomposition, axis=(2, 3)), (self.size_sat[1],self.size_sat[0]))
        decom_1 = decom_1 / np.max(decom_1)
        decom_2 = cv2.resize(np.sum(self.Decomposition, axis=(0, 1)), (self.size_grd[1],self.size_grd[0]))
        decom_2 = decom_2 / np.max(decom_2)

        image_overlay_1 = image_1 * 0.7 + self.imshow_convert(decom_1) / 255.0 * 0.3
        image_overlay_2 = image_2 * 0.7 + self.imshow_convert(decom_2) / 255.0 * 0.3

        plt.figure()
        plt.suptitle('Activation Decomposition (Overall map)')
        plt.subplot(2,2,1)
        plt.imshow(self.imshow_convert(decom_1))
        plt.subplot(2,2,2)
        plt.imshow(self.imshow_convert(decom_2))
        plt.subplot(2,2,3)
        plt.imshow(image_overlay_1)
        plt.subplot(2,2,4)
        plt.imshow(image_overlay_2)

        #--------------------------------------------------------------------------------
        # generate point-specific map, must generate the Decomposition first
        if self.Decomposition is None:
            self.Decomposition = self.Overall_map(map_1 = map_1, map_2 = map_2, mode = 'GAP')

        # query point, position in the feature matrix (not the x,y in image)
        query_point_1 = [260, 160]
        query_point_2 = [160, 340]

        # Use stream=1 for query point on image 1, the generated map is for image 2 (partial_2). vice versa
        partial_1 = self.Point_Specific(decom=self.Decomposition, point=query_point_2, stream=2, size=self.size_grd)
        partial_2 = self.Point_Specific(decom=self.Decomposition, point=query_point_1, stream=1, size=self.size_sat)

        partial_1 = cv2.resize(partial_1, (self.size_sat[1],self.size_sat[0]))
        partial_2 = cv2.resize(partial_2, (self.size_grd[1],self.size_grd[0]))
        partial_1 = partial_1 / np.max(partial_1)
        partial_2 = partial_2 / np.max(partial_2)

        image_overlay_1 = image_1 * 0.7 + self.imshow_convert(partial_1) / 255.0 * 0.3
        image_overlay_2 = image_2 * 0.7 + self.imshow_convert(partial_2) / 255.0 * 0.3

        plt.figure()
        plt.suptitle('Point-Specific Map')
        plt.subplot(2, 3, 1)
        plt.imshow(image_1)
        plt.plot(query_point_1[1], query_point_1[0], 'dr')
        plt.subplot(2, 3, 2)
        plt.imshow(self.imshow_convert(partial_2))
        plt.subplot(2, 3, 3)
        plt.imshow(image_overlay_2)
        plt.subplot(2, 3, 4)
        plt.imshow(image_2)
        plt.plot(query_point_2[1], query_point_2[0], 'dr')
        plt.subplot(2, 3, 5)
        plt.imshow(self.imshow_convert(partial_1))
        plt.subplot(2, 3, 6)
        plt.imshow(image_overlay_1)

def demo():
    generator = Explanation_generator()
    generator.demo()
    plt.show()

if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    demo()
