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
    last = 12
    for i in range(1,last):
        pcds_data.append(o3d.io.read_point_cloud("/Users/apple/Documents/SCAN 6 A POSE/"+str(i)+".pcd"))

    # pcds_data = body_rotator(pcds_data)

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
                                         
        # b1.transform(result_icp.transformation)
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
    
    # pcds_data = pcds_data[:5]



    print("all pcds with intial position",pcds_data)
    # a=pcds_data[:(last//2)+1]
    # ls=work(a)
    data = stich(pcds_data)
    o3d.visualization.draw_geometries([data])
    # print(test)

