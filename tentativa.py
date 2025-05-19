#!/usr/bin/env python

import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import PointCloud2

class PointCloudDifference:
    def __init__(self):
        rospy.init_node('removido')

        self.velodyne_cloud = None
        self.objects_cloud = None

        rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback)
        rospy.Subscriber('/pontos_filtrados', PointCloud2, self.objects_callback)

        self.pub = rospy.Publisher('/removido', PointCloud2, queue_size=1)

    def velodyne_callback(self, msg):
        self.velodyne_cloud = ros_numpy.numpify(msg)
        self.try_publish(msg.header)

    def objects_callback(self, msg):
        self.objects_cloud = ros_numpy.numpify(msg)
        self.try_publish(msg.header)

    def try_publish(self, header):
        if self.velodyne_cloud is None or self.objects_cloud is None:
            return

        points_velodyne = np.zeros((self.velodyne_cloud.shape[0], 3))
        points_velodyne[:, 0] = self.velodyne_cloud['x']
        points_velodyne[:, 1] = self.velodyne_cloud['y']
        points_velodyne[:, 2] = self.velodyne_cloud['z']

        points_objetos = np.zeros((self.objects_cloud.shape[0], 3))
        points_objetos[:, 0] = self.objects_cloud['x']
        points_objetos[:, 1] = self.objects_cloud['y']
        points_objetos[:, 2] = self.objects_cloud['z']

        
        dist_thresh = 2.0  # Margem de tolerância
        mask = np.ones(points_velodyne.shape[0], dtype=bool)

        for obj_pt in points_objetos:
            dist = np.linalg.norm(points_velodyne - obj_pt, axis=1)
            mask &= dist > dist_thresh  # Remove pontos próximos

        filtrados = self.velodyne_cloud[mask]
        msg_filtrado = ros_numpy.msgify(PointCloud2, filtrados, stamp=header.stamp, frame_id=header.frame_id)

        rospy.loginfo(f"Publicando nuvem filtrada: {np.count_nonzero(mask)} pontos restantes de {len(mask)}")
        self.pub.publish(msg_filtrado)

if __name__ == '__main__':
    try:
        PointCloudDifference()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
