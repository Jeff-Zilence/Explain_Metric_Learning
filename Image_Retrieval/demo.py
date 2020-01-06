import os,sys
from Model import BNInception
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Explanation_generator:

    def __init__(self):
        self.Decomposition = None

    # read image, subtract bias, convert to rgb for imshow
    def read(self, path):
        image = cv2.imread(path).astype(np.float)
        image_show = image[:,:,::-1]/255.
        image[:, :, 0] -= 104.
        image[:, :, 1] -= 117.
        image[:, :, 2] -= 128.
        return image, image_show

    # if you wanna load data from CUB or car dataset, you may download the dataset and write a simple dataloader.
    def get_input_from_path(self, path_1, path_2, size = (224,224)):
        '''
            load two images from paths
        '''
        inputs_1, image_1 = self.read(path_1)
        inputs_2, image_2 = self.read(path_2)

        image_1 = cv2.resize(image_1, (size[1], size[0]))
        image_2 = cv2.resize(image_2, (size[1], size[0]))

        inputs_1 = cv2.resize(inputs_1, (size[1], size[0]))
        inputs_2 = cv2.resize(inputs_2, (size[1], size[0]))

        inputs_1 = np.transpose(inputs_1, (2, 0, 1))
        inputs_2 = np.transpose(inputs_2, (2, 0, 1))
        # wrap them in Variable
        inputs_1 = Variable(torch.from_numpy(np.expand_dims(inputs_1.astype(np.float32), axis=0))).cuda()
        inputs_2 = Variable(torch.from_numpy(np.expand_dims(inputs_2.astype(np.float32), axis=0))).cuda()

        return inputs_1, image_1, inputs_2, image_2,

    def get_embed(self, inputs_1, inputs_2):
        '''
            The code is for situations where two models are different. 
            Since the models of two streams are the same in this case, 
            you may simplify the code for your project.
        '''
        model_1 = BNInception(dim=512)
        model_2 = BNInception(dim=512)

        # path to load the pretrained model
        resume = 'Image_Retrieval/Model/Multi_similarity.pth.tar' 

        # resume model
        print('load model from {}'.format(resume))
        chk_pt = torch.load(resume)
        weight = chk_pt['state_dict']
        start = chk_pt['epoch']
        model_1.load_state_dict(weight)
        model_2.load_state_dict(weight)

        model_1 = model_1.eval().cuda()
        model_2 = model_2.eval().cuda()

        embed_1, map_1, ori_1 = model_1(inputs_1, True)
        embed_2, map_2, ori_2 = model_2(inputs_2, True)

        fc_1 = model_1.classifier.linear.cpu()
        fc_2 = model_2.classifier.linear.cpu() 

        return embed_1, map_1, ori_1, fc_1, embed_2, map_2, ori_2, fc_2

    def imshow_convert(self, raw):
        '''
            convert the heatmap for imshow
        '''
        heatmap = np.array(cv2.applyColorMap(np.uint8(255*(1.-raw)), cv2.COLORMAP_JET))
        return heatmap

    def GradCAM(self, map, size = (224, 224)):
        gradient = map.grad.cpu().numpy()
        map = map.detach().cpu().numpy()

        # compute the average value
        weights = np.mean(gradient[0], axis=(1, 2), keepdims=True)
        grad_CAM_map = np.sum(np.tile(weights, [1, map.shape[-2], map.shape[-1]]) * map[0], axis=0)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = cv2.resize(cam, (size[1], size[0]))
        return cam

    def RGradCAM(self, map, size = (224, 224)):
        # rectified Grad-CAM, one variant
        gradient = map.grad.cpu().numpy()
        map = map.detach().cpu().numpy()

        # remove the heuristic GAP step
        weights = gradient[0]
        grad_CAM_map = np.sum(weights * map[0], axis = 0)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = cv2.resize(cam, (size[1], size[0]))
        return cam

    def Overall_map(self, map_1, map_2, fc_1 = None, fc_2 = None, size = (224, 224), mode = 'GMP'):
        '''
            Only for GMP architecture, you may check the code of other applications 
            for the implementation of GAP and flattened feature.
        '''
        if mode == 'GMP':
            map_1 = np.transpose(map_1.detach().cpu().numpy(),(0,2,3,1))
            map_2 = np.transpose(map_2.detach().cpu().numpy(),(0,2,3,1))

            # generate the maximum mask
            for k in range(map_1.shape[-1]):
                map_1[:,:,:,k] = map_1[:,:,:,k] * (map_1[:,:,:,k] == np.max(map_1[:,:,:,k]))
            for k in range(map_2.shape[-1]):
                map_2[:,:,:,k] = map_2[:,:,:,k] * (map_2[:,:,:,k] == np.max(map_2[:,:,:,k]))

            map_1_reshape = np.reshape(map_1, [-1, map_1.shape[-1]])
            map_2_reshape = np.reshape(map_2, [-1, map_2.shape[-1]])

            # compute the equivalent feature for each position
            map_1_embed = np.matmul(map_1_reshape, np.transpose(fc_1.weight.data.numpy())) #+ fc_1.bias.data.numpy() / map_1_reshape.shape[0]
            map_2_embed = np.matmul(map_2_reshape, np.transpose(fc_2.weight.data.numpy())) #+ fc_2.bias.data.numpy() / map_2_reshape.shape[0]
            map_1_embed = np.reshape(map_1_embed, [map_1.shape[1], map_1.shape[2],-1])
            map_2_embed = np.reshape(map_2_embed, [map_2.shape[1], map_2.shape[2],-1])

            Decomposition = np.zeros([map_1.shape[1],map_1.shape[2],map_2.shape[1],map_2.shape[2]])
            for i in range(map_1.shape[1]):
                for j in range(map_1.shape[2]):
                    for x in range(map_2.shape[1]):
                        for y in range(map_2.shape[2]):
                            Decomposition[i,j,x,y] = np.sum(map_1_embed[i,j]*map_2_embed[x,y])
            Decomposition = Decomposition / np.max(Decomposition)
            Decomposition = np.maximum(Decomposition, 0)
            return Decomposition

    def Point_Specific(self, decom, point = [0,0], stream = 1, size=(224,224)):
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

    def demo(self, path_1='Image_Retrieval/Images/51.jpg',
                   path_2='Image_Retrieval/Images/52.jpg',
                   size = (224,224)):
        '''
            Generate activation map with different methods.
            The model is based on global maximum pooling and trained with L2 normalization,
            so different variants have different results.
        '''
        inputs_1, image_1, inputs_2, image_2 = self.get_input_from_path(path_1=path_1, path_2=path_2, size=size)

        embed_1, map_1, ori_1, fc_1, embed_2, map_2, ori_2, fc_2 = self.get_embed(inputs_1=inputs_1, inputs_2=inputs_2)

        #--------------------------------------------------------------------------------
        # generate Grad-CAM
        map_1.retain_grad()
        map_2.retain_grad()

        product_vector = torch.mul(embed_1, embed_2)
        product = torch.sum(product_vector)
        product.backward(torch.tensor(1.).cuda(), retain_graph=True)

        gradcam_1 = self.GradCAM(map_1)
        gradcam_2 = self.GradCAM(map_2)

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

        #--------------------------------------------------------------------------------
        #Generate Grad-CAM using features without normalization (Grad-CAM_no_norm)
        map_1.retain_grad()
        map_2.retain_grad()

        product_vector = torch.mul(ori_1, ori_2)
        product = torch.sum(product_vector)
        product.backward(torch.tensor(1.).cuda(), retain_graph=True)

        gradcam_1 = self.GradCAM(map_1)
        gradcam_2 = self.GradCAM(map_2)

        image_overlay_1 = image_1 * 0.7 + self.imshow_convert(gradcam_1) / 255.0 * 0.3
        image_overlay_2 = image_2 * 0.7 + self.imshow_convert(gradcam_2) / 255.0 * 0.3

        plt.figure()
        plt.suptitle('Grad-CAM without Normalization')
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
            Generate overall activation map using activation decomposition,
            or the rectified Grad-CAM (RGrad-CAM)
            They are equivalent in certain situation, see the paper for details
        '''

        # compute the overall activation map with decomposition (no bias term)
        self.Decomposition = self.Overall_map(map_1 = map_1, map_2 = map_2, fc_1 = fc_1, fc_2 = fc_2, mode = 'GMP')

        decom_1 = cv2.resize(np.sum(self.Decomposition, axis=(2, 3)), (size[1], size[0]))
        decom_1 = decom_1 / np.max(decom_1)
        decom_2 = cv2.resize(np.sum(self.Decomposition, axis=(0, 1)), (size[1], size[0]))
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
        
        
        # generate the RGrad-CAM (equivalent with decomposition with bias term)
        map_1.retain_grad()
        map_2.retain_grad()

        product_vector = torch.mul(ori_1, ori_2)
        product = torch.sum(product_vector)
        product.backward(torch.tensor(1.).cuda())

        rgradcam_1 = self.RGradCAM(map_1)
        rgradcam_2 = self.RGradCAM(map_2)

        image_overlay_1 = image_1 * 0.7 + self.imshow_convert(rgradcam_1) / 255.0 * 0.3
        image_overlay_2 = image_2 * 0.7 + self.imshow_convert(rgradcam_2) / 255.0 * 0.3
        
        plt.figure()
        plt.suptitle('Rectified Grad-CAM (Decomposition+Bias)')
        plt.subplot(2, 2, 1)
        plt.imshow(self.imshow_convert(rgradcam_1))
        plt.subplot(2, 2, 2)
        plt.imshow(self.imshow_convert(rgradcam_2))
        plt.subplot(2, 2, 3)
        plt.imshow(image_overlay_1)
        plt.subplot(2, 2, 4)
        plt.imshow(image_overlay_2)
        '''
        '''
        #--------------------------------------------------------------------------------
        # generate point-specific map, must generate the Decomposition first
        if self.Decomposition is None:
            self.Decomposition = self.Overall_map(map_1 = map_1, map_2 = map_2, fc_1 = fc_1, fc_2 = fc_2, mode = 'GMP')

        # query point, position in the feature matrix (not the x,y in image)
        query_point_1 = [100, 128] 
        query_point_2 = [100, 128] 

        # Use stream=1 for query point on image 1, the generated map is for image 2 (partial_2). vice versa
        partial_1 = self.Point_Specific(decom=self.Decomposition, point=query_point_2, stream=2)
        partial_2 = self.Point_Specific(decom=self.Decomposition, point=query_point_1, stream=1)

        partial_1 = cv2.resize(partial_1, (size[1], size[0]))
        partial_2 = cv2.resize(partial_2, (size[1], size[0]))
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
