import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence
#添加一个保存图像的
from torchvision.utils import save_image

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        
        #
        grid_image1 = grid_image
        writer.add_image('Image', grid_image, global_step)


        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(), dataset=dataset), 3, normalize=False, range=(0, 255))
        
        #
        grid_image2 = grid_image  
        writer.add_image('Predicted label', grid_image, global_step)
       
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(), dataset=dataset), 3, normalize=False, range=(0, 255))
        
        #
        grid_image3 = grid_image  
        writer.add_image('Groundtruth label', grid_image, global_step)


        if 1:
            ############################################################################
            # #我们在这里保存一下图像到本地，因为服务器不可以可视化 TODO
            # fake_image_list = []
            # # coarse_map = self.FCN8(fixed_x)
            # # refined_map= self.guidance_module(fixed_x,coarse_map)

            # lbl_pred = output.data.max(1)[1].cpu().numpy()
            # #lbl_pred_refined = refined_map.data.max(1)[1].cpu().numpy()
            # lbl_pred = self.data_loader.dataset.colorize_mask_batch(lbl_pred)
            # #lbl_pred_refined = self.data_loader.dataset.colorize_mask_batch(lbl_pred_refined)
            # lbl_true = self.data_loader.dataset.colorize_mask_batch(fixed_target.numpy())
            # # print(lbl_pred.size()) 
            # # print(lbl_pred_refined.size()) 
            # # print(lbl_true.size()) 
            # fake_image_list.append(lbl_pred)
            # #fake_image_list.append(lbl_pred_refined)
            # fake_image_list.append(lbl_true)
            # # fake_image_list.append(lbl_pred_refined.unsqueeze(1).expand(fixed_x.size()).float())
            # # fake_image_list.append(lbl_true)
            # fake_images = torch.cat(fake_image_list, dim=3)

            # save_image(grid_image1,
            #     os.path.join('/fast/users/a1746546/code/pytorch-deeplab-xception/run/pascal/deeplab-resnet/imagesRs', 
            #     '{}_SrcImg.png'.format(global_step)),nrow=1, padding=0)
            
            # save_image(grid_image2,
            #     os.path.join('/fast/users/a1746546/code/pytorch-deeplab-xception/run/pascal/deeplab-resnet/imagesRs', 
            #     '{}_SrcPred.png'.format(global_step)),nrow=1, padding=0)
            

            # save_image(grid_image3,
            #     os.path.join('/fast/users/a1746546/code/pytorch-deeplab-xception/run/pascal/deeplab-resnet/imagesRs', 
            #     '{}_SrcImgGt.png'.format(global_step)),nrow=1, padding=0)
            
            # print('Translated images and saved into {}..!'.format(self.sample_path))

            # del coarse_map, refined_map, lbl_pred, lbl_pred_refined, fake_image_list 
            
            #########################################################################    
            