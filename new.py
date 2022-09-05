import open3d as o3d
import numpy as np
import copy
import os

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
    source.orient_normals_consistent_tangent_plane(k=125)
    target.estimate_normals()
    target.orient_normals_consistent_tangent_plane(k=125)
    
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    
    source.transform(trans_init)
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


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


if __name__ == "__main__":

    voxel_size = 0.001  # means 5cm for the dataset
    
    t=[]
    dir = os.listdir("/Users/apple/Documents/SCAN 2")
    dir.sort()
    pcds_data = []
    for i in dir:
        pcds_data.append(o3d.io.read_point_cloud("/Users/apple/Documents/SCAN 2/"+i))


    print(pcds_data)
    ls = pcds_data
    a = 150
    c=1
    for i in range(len(pcds_data)-1):
        print("lala",len(ls))
        temp = []
        for j in range(len(ls)-1):
            
            source = ls[j]
            target = ls[j+1]
            source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(voxel_size,source,target)
            result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
            print(result_ransac)
            val = np.asarray(result_ransac.correspondence_set)
            while val.shape[0] < a:
                result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
                print(result_ransac)
                # draw_registration_result(source_down, target_down,
                                        # result_ransac.transformation)
                val = np.asarray(result_ransac.correspondence_set)

            result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                                voxel_size)
            # print(result_icp)
            source_temp = copy.deepcopy(source)
            source_temp.transform(result_icp.transformation)
            temp.append(source_temp + target)
            o3d.io.write_point_cloud("result/"+str(c)+".ply",source_temp + target)
            c+=1
            o3d.visualization.draw_geometries([source_temp + target])
            # print(temp)
        a+=50
        ls = temp
        
        
    print(ls)
    # o3d.io.write_point_cloud("result",ls[0])
    o3d.visualization.draw_geometries([ls[0]])
    # draw_registration_result(source, target, result_icp.transformation)
#     RegistrationResult with fitness=8.160023e-01, inlier_rmse=8.658453e-03, and correspondence_set size of 23110
# Access transformation to get result.
# [PointCloud with 41498 points.]
# [PointCloud with 41498 points.]