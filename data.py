import os
import glob
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import random
from tqdm import tqdm
import open3d as o3d
import random
import shutil


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    #xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    #xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
      
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class Numbers(Dataset):
    def __init__(self, num_points, data_path, partition='train', xyz_unit=20,
                 index_list=None, save_file_name='processing_data.h5', 
                 make_select=True, normal_probability=0.5, random_seed=42,
                 save_path=None, train_serial=None, label_save=False, 
                 sampling_method='normal', model_sampling=False,
                  voxel_downsampling=False, voxel_size=0.02, 
                  outlier_remover=False, nb_points=16, radius=0.05,
                  weight_method='list'
                 ):
        """
        sampling_method
        - normal : 평소에 쓰는 방법
        - random : random으로 sampling -> voxel_downsampling, model_sampling 등에서 쓰일 듯
        """
        
        self.xyz_unit = xyz_unit
        self.num_points = num_points
        self.partition = partition     
        self.save_file_name = f'{partition}_data.h5'
        self.random_seed = random_seed
        self.normal_probability = normal_probability
        self.weight_method = weight_method
        self.random_weight = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
        
        self.save_path = save_path 
        self.label_save = label_save
        
        self.sampling_method = sampling_method
        self.model_sampling = model_sampling
        self.voxel_downsampling = voxel_downsampling
        self.voxel_size = voxel_size
        self.outlier_remover = outlier_remover
        self.nb_points = nb_points
        self.radius = radius


        
        train_df = pd.read_csv(data_path+'/train.csv')
        
        if train_serial:
            train_path = f'./result/train/{train_serial}'
            self.X_points = h5py.File(f'{train_path}/{self.save_file_name}', 'r')
            
            self.ids = [key for key in self.X_points.keys()]
            if partition!='test':
                target_inds = train_df[train_df['ID'].isin([int(key) for key in self.ids])].index.tolist()
                self.ids = train_df.loc[target_inds,'ID'].tolist()
                self.ids = [str(i) for i in self.ids]
                self.label = train_df.loc[target_inds,'label'].tolist()

        else:
            if partition!='test':
                
                
                all_points = h5py.File(f'{data_path}/train.h5', 'r')
                if index_list:
                    train_df = train_df.loc[index_list,:].reset_index(drop=True)
                self.ids = train_df['ID'].tolist()
                self.label = train_df['label'].tolist()
                
                self.ids = [str(i) for i in self.ids]
            else:
                all_points = h5py.File(f'{data_path}/test.h5', 'r')
                self.ids = [k for k in all_points.keys()]
                
            h5_save_path = f'{data_path}/{self.save_file_name}'
            
            if make_select:
                self.select_points(self.ids, all_points, h5_save_path)
            all_points.close()
            self.X_points = h5py.File(h5_save_path, 'r')
            
            if self.save_path:
                copy_path = os.path.join(self.save_path, self.save_file_name)
                shutil.copyfile(h5_save_path, copy_path)
                if self.label_save:
                    with open(f'{self.save_path}/{self.partition}_label.pkl', 'wb') as f:
                        pickle.dump(self.label, f)
                    with open(f'{self.save_path}/{self.partition}_id.pkl', 'wb') as f:
                        pickle.dump(self.ids, f)

    def __getitem__(self, item):
        #pointcloud = self.data[item][:self.num_points]
        pointcloud = np.array(self.X_points[self.ids[item]])

        if self.partition=='train':
            p = np.random.rand()
            if p>self.normal_probability:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointcloud)
                if self.weight_method=='list':
                    
                    x_w = random.choice(self.random_weight)
                    y_w = random.choice(self.random_weight)
                    z_w = random.choice(self.random_weight)
                else:
                    x_w,y_w,z_w = np.random.uniform(0,2,3)
                
                R = pcd.get_rotation_matrix_from_xyz((x_w*np.pi, y_w*np.pi, z_w*np.pi))
                pointcloud = np.asarray(pcd.rotate(R, center=(0,0,0)).points)

        if self.partition!='test':
            label = np.array(int(self.label[item]))
        

        if self.partition == 'train' and self.model_sampling['str']:
            if self.model_sampling['dropout']:
                pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            if self.model_sampling['translate']:
                pointcloud = translate_pointcloud(pointcloud)
            if self.model_sampling['shuffle']:
                np.random.shuffle(pointcloud)

        pointcloud = pointcloud.astype('float32')

        if self.partition!='test':
            return pointcloud, label
        else:
            return pointcloud
    
    
    def make_position_dict(self, origin_point_obj, ind, length, segments):
        
        points_obj = sorted(origin_point_obj, key=lambda x:x[1][ind])
        
        n = 0
        v_dict = {}
        v_start = segments[ind][n]
        v_end = segments[ind][n+1]
        for i, point in points_obj:
            v = point[ind]
    
            if v>v_start and v<=v_end:
                v_dict[i] = n
            else:
                while n<length-1:
                    n += 1
                    v_start = segments[ind][n]
                    v_end = segments[ind][n+1]
                    if v>v_start and v<=v_end:
     
                        v_dict[i] = n
                        break
        return v_dict
    
    
    def select_points(self, ids, all_points, save_path):
        
        xyz_list = [self.xyz_unit,self.xyz_unit,self.xyz_unit]

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        count = 0
        h5f = h5py.File(save_path, 'w')
        for key in tqdm(ids):
            final_points = []
            final_cand_points = []
            points = np.array(all_points[key])

            if self.outlier_remover:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                #uni_down_pcd = pcd.uniform_down_sample(every_k_points=2)
                voxel_down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=self.nb_points, radius=self.radius)
                points = np.asarray(cl.points)
                del pcd
                
            if self.voxel_downsampling:
            
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                points = np.asarray(downpcd.points)
                del pcd, downpcd
            
            if self.sampling_method=='normal':
    
                n = points.shape[0]
                points_obj = [(i, point) for i, point in zip(range(len(points)), points)]
            
                xyz_min = np.min(points, axis=0)
                xyz_max = np.max(points, axis=0)
                
                eps = 1e-6
                
                xyz_min = xyz_min - eps
                xyz_max = xyz_max + eps
                
                segments = []
                for i in range(3):
                    values = np.linspace(xyz_min[i], xyz_max[i], num=xyz_list[i]+1)
                    segments.append(values)
                
                x_dict = self.make_position_dict(points_obj, 0, self.xyz_unit, segments)
                y_dict = self.make_position_dict(points_obj, 1, self.xyz_unit, segments)
                z_dict = self.make_position_dict(points_obj, 2, self.xyz_unit, segments)
                
                for i in range(self.xyz_unit):
                    t1s = []
                    for j in range(self.xyz_unit):
                        t2s = []
                        for k in range(self.xyz_unit):
                            t2s.append([])
            
                        t1s.append(t2s)
                    final_points.append(t1s)
            
                sample_num = 0
            
                for ind in x_dict.keys():
                    x_num = x_dict[ind]
                    y_num = y_dict[ind]
                    z_num = z_dict[ind]
                    final_points[x_num][y_num][z_num].append(ind)

                # select inds
                sample_inds = []
                second_cand_inds = []
                sample_num = 0
                
                sample_ok_num_set = []
                for i in range(self.xyz_unit):
                    for j in range(self.xyz_unit):
                        for k in range(self.xyz_unit):
                            cand_inds = final_points[i][j][k]
                            if cand_inds:
            
                                one_selected_cand = random.choice(cand_inds)
            
                                sample_inds.append(one_selected_cand)
                                sample_num += 1
                                for cand_ind in cand_inds:
                                    if cand_ind == one_selected_cand:
                                        continue
                                    second_cand_inds.append(cand_ind)
                                
                                if len(cand_inds)>=20:
                                    sample_ok_num_set += [(i,j,k)]*len(cand_inds)
                
                last_points = np.array([])
                if self.num_points<sample_num:
                    sample_inds = random.sample(sample_inds, self.num_points)
    
                else:
                    rest_num = self.num_points-sample_num
                    
                     
                    if rest_num<=len(second_cand_inds):
                        sample_inds += random.sample(second_cand_inds, rest_num)
                    else:
                        if len(sample_ok_num_set)==0:
                            import pdb
                            pdb.set_trace()
                        temp_length = len(second_cand_inds)
                        sample_inds += random.sample(second_cand_inds, temp_length)
                        rest_num -= len(second_cand_inds)
                        
                        
                        #temp = np.random.choice(sample_ok_num_set, size=rest_num, replace=True)
                        
                        sample_ok_num_dict = {}
                        for _ in range(rest_num):
                            sample_tuple = random.choice(sample_ok_num_set)
                            if sample_tuple not in sample_ok_num_dict.keys():
                                sample_ok_num_dict[sample_tuple] = 0
                            sample_ok_num_dict[sample_tuple] += 1
                            
                        
                        for key in sample_ok_num_dict.keys():
                            i,j,k = key
                            t_num = sample_ok_num_dict[key]
                            temp_inds = final_points[i][j][k]
                            temp_points = points[temp_inds]
                            
                            t_samples = np.random.multivariate_normal(np.mean(temp_points, axis=0), np.cov(temp_points.T), t_num)
                            if len(last_points)==0:
                                last_points = t_samples
                            else:
                                
                                last_points = np.r_[last_points,t_samples]
                    
                
                final_cand_points = points[sample_inds]
                if len(last_points)>0:
                    final_cand_points = np.r_[final_cand_points, last_points]
            
            
            
            elif self.sampling_method=='random':

                inds = np.random.choice([i for i in range(points.shape[0])], size=self.num_points, replace=False)
                points = points[inds,:]
                
            
            if self.partition=='valid':
                p = np.random.rand()
                if p>self.normal_probability:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(final_cand_points)

                    if self.weight_method=='list':
                    
                        x_w = random.choice(self.random_weight)
                        y_w = random.choice(self.random_weight)
                        z_w = random.choice(self.random_weight)
                    else:
                        x_w,y_w,z_w = np.random.uniform(0,2,3)
                    
                    R = pcd.get_rotation_matrix_from_xyz((x_w*np.pi, y_w*np.pi, z_w*np.pi))
                    final_cand_points = np.asarray(pcd.rotate(R, center=(0,0,0)).points)
            
            if not final_cand_points:
                final_cand_points = points
            

            if final_cand_points.shape[0]>self.num_points:
                raise ValueError(f"current points are more than num_points ({final_cand_points.shape[0]}>{self.num_points})")
            elif final_cand_points.shape[0]<self.num_points:
                raise ValueError(f"current points are more less num_points ({final_cand_points.shape[0]}<{self.num_points})")
            #self.pointcloud.append(final_cand_points)
            h5f.create_dataset(key, data=final_cand_points)
            count += 1
            """
            if count == 100:
                break
            """
        h5f.close()
        
        #self.pointcloud = np.array(self.pointcloud)

    
    def __len__(self):
        return len(self.ids)

