#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelStates
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sm

class HybridBoxFilterTF:
    def __init__(self):
        rospy.init_node('filter_tf_node')

        self.model_size = np.array([1.5, 1.5, 1.5])  
        self.ignored_models = ['ground_plane', 'robot', 'inspector'] 

        self.model_poses = {}  

     
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

      
        self.sub_lidar = rospy.Subscriber("/velodyne_points", PointCloud2, self.lidar_callback)
        self.sub_gazebo = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)

       
        self.pub_filtered = rospy.Publisher("/pontos_filtrados", PointCloud2, queue_size=1)
        self.pub_debug_all = rospy.Publisher("/pontos_lidar", PointCloud2, queue_size=1)

        rospy.loginfo("Filtro iniciado.")

    def model_states_callback(self, msg):
        
        self.model_poses.clear()
        for name, pose in zip(msg.name, msg.pose):
            if any(skip in name for skip in self.ignored_models):
                continue
            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            self.model_poses[name] = (pos, pose.orientation)
            rospy.loginfo_throttle(10, f"Modelo válido detectado: {name}, posição: {pos}")
        if not self.model_poses:
            rospy.logwarn_throttle(5, "Nenhum modelo válido encontrado para filtragem.")

    def lidar_callback(self, msg):

        if not self.model_poses:
            return

        try:
            transform = self.tf_buffer.lookup_transform("base_link", msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            cloud_transformed = tf2_sm.do_transform_cloud(msg, transform)
        except Exception as e:
            rospy.logwarn(f"Erro ao transformar nuvem para 'base_link': {e}")
            return

        points = np.array(list(pc2.read_points(cloud_transformed, field_names=("x", "y", "z"), skip_nans=True)))
        if len(points) == 0:
            rospy.logwarn("Nuvem de pontos vazia após transformação.")
            return

        self.pub_debug_all.publish(cloud_transformed)

        all_filtered_points = []

        for name, (pos, orientation) in self.model_poses.copy().items():

            center = pos + np.array([0, 0, self.model_size[2] / 2])
            lower = center - self.model_size / 2.0
            upper = center + self.model_size / 2.0

            mask = np.all((points >= lower) & (points <= upper), axis=1)
            filtered = points[mask]
            if len(filtered) > 0:
                all_filtered_points.append(filtered)
                rospy.loginfo_throttle(5.0, f"{len(filtered)} pontos dentro da bounding box de '{name}'.")

        if not all_filtered_points:
            return

        combined = np.vstack(all_filtered_points)
        filtered_msg = self.create_pointcloud2(cloud_transformed.header, combined)
        self.pub_filtered.publish(filtered_msg)

    def create_pointcloud2(self, header, points):

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        return pc2.create_cloud(header, fields, points)

if __name__ == "__main__":
    try:
        HybridBoxFilterTF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
