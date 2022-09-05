from cmath import cos
import open3d as o3d
import numpy as np
import copy
import os
import math

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    #o3d.visualization.draw_geometries([source_temp, target_temp], width=1024, height=768)
    
    
    newpointcloud = source_temp + target_temp 
    #o3d.visualization.draw_geometries(newpointcloud)
    o3d.visualization.draw_geometries([newpointcloud], width=1024, height=768)
    o3d.io.write_point_cloud ( "result.ply", newpointcloud)


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # o3d.visualization.draw_geometries([pcd_down])
    print("Down______",pcd_down)
    # pcd_down = pcd
    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=300))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size,source,target):
    # print(":: Load two point clouds and disturb initial pose.")
    
    source.estimate_normals()
    # o3d.visualization.draw_geometries([source])
    source.orient_normals_consistent_tangent_plane(k=25)
    # o3d.visualization.draw_geometries([source])
    target.estimate_normals()
    target.orient_normals_consistent_tangent_plane(k=25)
    
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], 
                            [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

# RANSAC implementation 
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True , distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, result_ransac.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # return result
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,result_ransac.transformation ,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def stich(arr):
    result = arr[0]
    for i in ls[1:]:
        result = result + i
    return result


def work(pcds_data):
    ls = pcds_data
    a = []
    c=0
    # tran_data={}
    test=[]
    ab=0
    # mat = arr[0]
    # for i in range(len(pcds_data)-1):
    for i  in range(1,len(pcds_data)-1):
        
        print("frame: ",i)
        trg = copy.deepcopy(ls[i-1])
        src = copy.deepcopy(ls[i])
        arr = [[[0.866,0.,0.5,0.],[0.,1.,0.,0.],[-0.5,0.,0.8660,0],[0.,0.,0.,1.]],
        
        [[0.5,0.,0.8660,0],[0.,1.,0.,0.],[-0.866,0.,0.5,0.],[0.,0.,0.,1.]],

[[0,0.,1,0.],[0.,1.,0.,0.],[-1,0.,0,0],[0.,0.,0.,1.]],

[[-0.5,0.,0.8660,0],[0.,1.,0.,0.],[-0.866,0.,-0.5,0.],[0.,0.,0.,1.]],


[[-0.866,0.,0.5,0.],[0.,1.,0.,0.],[-0.5,0.,-0.8660,0],[0.,0.,0.,1.]],


[[-1,0.,0.,0.],[0.,1.,0.,0.],[0,0.,-1,0],[0.,0.,0.,1.]],

[[-0.866,0.,-0.5,0.],[0.,1.,0.,0.],[0.5,0.,-0.8660,0],[0.,0.,0.,1.]],

[[-0.5,0.,-0.8660,0],[0.,1.,0.,0.],[0.866,0.,-0.5,0.],[0.,0.,0.,1.]],

[[0,0.,-1,0.],[0.,1.,0.,0.],[1,0.,0,0],[0.,0.,0.,1.]],

[[0.5,0.,-0.8660,0],[0.,1.,0.,0.],[0.866,0.,0.5,0.],[0.,0.,0.,1.]],

[[0.866,0.,-0.5,0.],[0.,1.,0.,0.],[0.5,0.,0.8660,0],[0.,0.,0.,1.]],

[[1,0.,0.,0.],[0.,1.,0.,0.],[0,0.,1,0],[0.,0.,0.,1.]]]

        src.transform(arr[i-1])
        source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(voxel_size,src,trg)

        result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)

        print(result_ransac)

        val = np.asarray(result_ransac.correspondence_set)

        score = 500
        while val.shape[0] < score  :
            result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
            print(result_ransac)

            val = np.array(result_ransac.correspondence_set)
            # print(val)
            if ab < val.shape[0]:
                ab = val.shape[0]
            c+=1
            if c > 500:
                score = ab
                c=0
                ab=0

        a1,b1 = copy.deepcopy(ls[i]),copy.deepcopy(ls[i])   
        a1.paint_uniform_color([1, 0.706, 0])
        b1.paint_uniform_color([0, 0.651, 0.929])
        a1.transform(result_ransac.transformation)       
        ab=0
        a.append(val.shape[0])
        test.append(score)

        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                            voxel_size,result_ransac)
                                         
        b1.transform(result_icp.transformation)
        # o3d.visualization.draw_geometries([a1,b1,src])
        # print(result_icp)
        # source_temp = copy.deepcopy(source)
        # o3d.visualization.draw_geometries([source , target])
        
        # data = copy.deepcopy(np.asarray(result_icp.transformation))
        
        # mat[0][3] = data[0][3]
        # mat[1][3] = data[1][3]
        # mat[2][3] = data[2][3]
        # mat = np.asarray(mat)
        # # print(result_icp.transformation)
        # print(mat)
        
        ls[i]=src.transform(result_icp.transformation)
        # pcds_data[i] = source.transform(np.ndarray(mat))
        # tran_data[str(i+1)]=result_icp.transformation
        
        # temp.append(source_temp + target)
        # o3d.io.write_point_cloud("result/"+str(c)+".ply",source_temp + target)
        # c+=1
        # o3d.visualization.draw_geometries([source_temp + target])
    return ls

def body_rotator(arr):
    n = len(arr)
    angle = 360 / n
    trans_mat = [[math.cos(angle),0.,math.sin(angle),0.],[0.,1.,0.,0.],[-math.sin(angle),0.,math.cos(angle),0],[0.,0.,0.,1.]]
    for i in range(n):
        c=0
        while c == i:
            arr[i] = arr[i].transform(trans_mat)
            c+=1
    return arr

if __name__ == "__main__":

    voxel_size = 0.025 # means 5cm for the dataset
    
    t=[]
    # dir = os.listdir("/Users/apple/Documents/new")
    # dir.sort()
    # print(dir)
    pcds_data = []
    last = 19
    for i in range(1,last):
        pcds_data.append(o3d.io.read_point_cloud("/Users/apple/Documents/new4/"+str(i)+".pcd"))

    pcds_data = body_rotator(pcds_data)

    """ arr = [ 
    [[ 0.09480523,  0.99458928, 0.0424751 ,  0],
 [-0.43585337 , 0.00310989,  0.90001231 , 0],
 [ 0.89501051 ,-0.10383879 , 0.43378993 , 0],
 [ 0.       ,   0.    ,      0.       ,   1.       ]]
,

[[ 0.4571747  , 0.86250282 , 0.21697969 , 0],
 [-0.44234936, 0.00886231 , 0.89679903  ,0.],
 [ 0.77156875, -0.50597465 , 0.38557919,  0],
 [ 0.     ,     0.       ,   0.     ,     1.        ]]
,
[[ 0.79610502  ,0.48035159  ,0.36807492 , 0],
 [-0.44172574 , 0.04552291  ,0.89599444 , 0],
 [ 0.41363651, -0.87589384 , 0.24842468 ,0],
 [ 0.     ,     0.     ,     0.     ,     1.        ]]
,
[[ 0.90855982  ,0.06806971 , 0.41217178 , 0],
 [-0.41602458,  0.05772555 , 0.90751932,  0],
 [ 0.03798173 ,-0.99600917 , 0.08076579 ,0],
 [ 0.      ,    0.        ,  0.       ,   1.        ]]
,
[[ 0.85286262,-0.33562363 , 0.39997766 , 0],
 [-0.40616835 , 0.05491981 , 0.91214642 , 0],
 [-0.32810458, -0.94039385, -0.08948064 ,0],
 [ 0.       ,   0.        ,  0.    ,      1.        ]]
,
[[ 0.54598851 ,-0.8022242 ,  0.24152201,  0],
 [-0.37920101,  0.02043057  ,0.92508875 , 0],
 [-0.74706301 ,-0.59667321 , -0.29304937 , 0 ],
 [ 0.    ,      0.      ,    0.      ,    1.        ]]
,
[[ 0.04953951 ,-0.99875859 ,-0.00520783 ,0],
 [-0.3765898 , -0.02350814,  0.9260818 ,  0],
 [-0.92505457 ,-0.04391643, -0.37728688, 0],
 [ 0.       ,   0.     ,     0.    ,      1.        ]]
,
[[-0.2914836 , -0.94184753 ,-0.16721405 ,0],
 [-0.39546423 ,-0.04051837 , 0.91758722 , 0],
 [-0.8710025 ,  0.3335888  ,-0.36065657 ,0],
 [ 0.       ,   0.      ,    0.        ,  1.        ]]
, 
[[-0.61866565 ,-0.72113419 ,-0.3117985 , 0],
 [-0.41165929 ,-0.04048318  ,0.91043821 , 0],
 [-0.66917071 , 0.6916116 , -0.27181604, -0.22263234],
 [ 0.       ,   0.     ,     0.    ,      1.        ]]
,
[[-0.8213059 , -0.38359829 ,-0.42226647, 0],
 [-0.4363214 , -0.05448648 , 0.89813966 , 0],
 [-0.36753265 , 0.9218913 , -0.12262207, 0],
 [ 0.       ,   0.     ,     0.      ,    1.        ]]
,
[[-0.886983  ,  0.06720214 ,-0.45688623 ,0],
 [-0.46029425 ,-0.04876984 , 0.8864258 ,  0],
 [ 0.03728744 , 0.99654672  ,0.07419078 ,0],
 [ 0.       ,   0.    ,      0.       ,   1.        ]]
]"""
   
    
    
    # pcds_data = pcds_data[:5]




    print("all pcds with intial position",pcds_data)
    a=pcds_data[:(last//2)+1]
    # ls=work(a)
    data = stich(pcds_data)
    o3d.visualization.draw_geometries([data])
    # print(test)

