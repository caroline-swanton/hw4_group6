'''
This node takes the laser scan from the Lidar and transforms
it into a 2D point cloud, then calculates the transform via
matched points, and then merges the new points with the existing cloud
'''
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from laser_geometry import LaserProjection
import tf2_ros
from tf2_ros import TransformException, TransformStamped  # pylint: disable=no-name-in-module
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial import cKDTree
from tf_transformations import euler_from_quaternion

# pylint: disable=too-many-instance-attributes
class PauseAndCapture(Node):
    """ This defines the scan_match node to take in Lidar points"""
    def __init__(self):
        super().__init__('pause_and_capture')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.laser_projector = LaserProjection()

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)

        # Set up the subscription for LaserScan message
        # HINT: Subscribe on the '/scan' topic
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile)

        # Create a publisher for PointCloud2 messages
        # HINT: Publish on the '/accumulated_cloud' topic
        self.pc_pub = self.create_publisher(PointCloud2, '/accumulated_cloud', 10)

        # Create a publisher for ICP merged cloud
        # HINT: Publish on the '/icp_merged_cloud' topic
        self.icp_pub = self.create_publisher(PointCloud2, '/icp_merged_cloud', 10)

        self.accumulated_points = []
        self.icp_accumulated_points = []

        self.capture_enabled = False
        self.latest_scan = None
        self.delay_timer = None

        self.input_thread = threading.Thread(target=self.key_press_listener, daemon=True)
        self.input_thread.start()

        self.get_logger().info("PauseAndCapture node started. Press Enter to capture a scan.")

    def key_press_listener(self):
        """Listens for terminal input"""
        while True:
            input(">> Press Enter to capture scan: ")
            self.capture_enabled = True

    def scan_callback(self, scan_msg: LaserScan):
        """Callback for Lidar data"""
        if not self.capture_enabled:
            return

        self.latest_scan = scan_msg
        self.capture_enabled = False

        if self.delay_timer:
            self.delay_timer.cancel()
        self.delay_timer = self.create_timer(0.1, self.delayed_transform_lookup)

    def delayed_transform_lookup(self):
        """Looks up transform between clouds"""
        self.delay_timer.cancel()
        scan_msg = self.latest_scan
        self.latest_scan = None

        try:
            cloud_in_laser = self.laser_projector.projectLaser(scan_msg)

            # Perform a lookup to transform the point cloud from its original
            # frame to the 'odom' frame
            transform = self.tf_buffer.lookup_transform(
                'odom',  # Target frame (where do you want to transform to?)
                scan_msg.header.frame_id,  # Source frame (the cloud's original frame)
                # Use the scan's timestamp for proper time synchronization
                rclpy.time.Time.from_msg(scan_msg.header.stamp),
                # Wait up to 0.5 seconds for the transform
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            # Transform the point cloud with the transform_pointcloud2 function
            transformed_points = self.transform_pointcloud2(cloud_in_laser, transform)

            if self.icp_accumulated_points:
                icp_aligned = self.perform_icp(self.icp_accumulated_points, transformed_points)
                self.icp_accumulated_points.extend(icp_aligned)
                self.publish_icp_merged_cloud(scan_msg.header.stamp)
                self.get_logger().info(f"ICP-aligned and merged {len(icp_aligned)} points.")
            else:
                self.icp_accumulated_points.extend(transformed_points)
                self.publish_icp_merged_cloud(scan_msg.header.stamp)
                self.get_logger().info(
                    f"Initialized ICP merged cloud with {len(transformed_points)} points.")

            self.accumulated_points.extend(transformed_points)
            self.publish_accumulated_cloud(scan_msg.header.stamp)
            self.get_logger().info(f"Captured and transformed {len(transformed_points)} points.")

        except TransformException as ex:
            self.get_logger().warn(f"Transform failed after delay: {str(ex)}")

    # Note that ros inherently processes point clouds in 3d
    # even though the robot's point cloud is in 2d.
    def transform_pointcloud2(self, cloud_msg: PointCloud2, transform: TransformStamped) \
            -> list[tuple[float, float, float]]:
        """Transform a point cloud using Euler angles from a given quaternion."""
        # pylint: disable=invalid-name,too-many-locals,too-many-arguments

        def rotate_point_euler(x, y, z, roll, pitch, yaw) -> tuple[float, float, float]:
            """Rotate a point (x, y, z) using Euler angles (roll, pitch, yaw)."""
            # Using the roll, pitch and yaw construct the Rx, Ry, Rz matrix
            R_x = np.array([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])
            R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])
            R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])

            # Combined rotation matrix
            R_tot = R_z @ R_y @ R_x

            # Apply the rotation to the point
            result = R_tot @ np.array([x, y, z])

            return tuple(result)

        # Extract translation and rotation (quaternion) from the transform method
        trans = transform.transform.translation
        rot = transform.transform.rotation
        quat = [rot.x, rot.y, rot.z, rot.w]  # tf_transformations wants x,y,z,w

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Hint: Use the euler_from_quaternion
        (roll, pitch, yaw) = euler_from_quaternion(quat)

        # Transform the point cloud using Euler rotation
        transformed_points = []
        for pt in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            # Get values of pt
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])

            # Apply rotation to the point using the rotate_point_euler function
            (tx, ty, tz) = rotate_point_euler(x, y, z, roll, pitch, yaw)

            # Append transformed point
            transformed_points.append((tx + trans.x, ty + trans.y, tz + trans.z))

        return transformed_points

    def perform_icp(self, previous_points, current_points, max_iterations=20, tolerance=1e-4):
        """Main ICP loop to transform new points"""
        # pylint: disable=invalid-name,too-many-locals
        src = np.array(current_points)
        tgt = np.array(previous_points)

        # Useful Steps to Follow For the Loop:
        # 1. Start icp loop for max_iterations
        # 2. Build KDTree for target cloud, cKDTree from scipy
        # 3. Find nearest neighbors from source to tgt
        # 4. Compute centroids of matched source and target points
        # 5. Center both point clouds by subtracting their centroids
        # 6. Compute the cross-covariance matrix
        # 7. SVD on step 6
        # 8. Compute rotation matrix R from SVD, np.linalg.svd will help
        # 9. Compute translation vector t from centroids and rotation
        # 10. Apply the transformation to the source points
        # 11. Compute mean error and check for convergence
        # 12. If converged, break the loop

        tgt_kdtree = cKDTree(tgt)  # KDTree for target cloud

        for _ in range(max_iterations):
            # Find nearest neighbors from source to target
            distances, indices = tgt_kdtree.query(src)
            matched_tgt = tgt[indices]

            # SVD estimation of the best-fit rotation
            U, _, V_T = self.svd_estimation(matched_tgt, src)
            R = V_T.T @ U.T

            # Translation from centroids
            src_centroid = np.mean(src, axis=0)
            tgt_centroid = np.mean(matched_tgt, axis=0)
            t = tgt_centroid - R @ src_centroid

            # Apply transformation
            src = (R @ src.T).T + t

            E = np.mean(distances ** 2)  # mean squared error
            if E < tolerance:  # check for convergence
                break

        return src.tolist()

    # Curr = Source, Prev = Target
    def svd_estimation(self, previous_points, current_points):
        """Calculates matrices for U, V_T, and Sigma"""
        # pylint: disable=invalid-name
        # compute centroids
        src_centroid = np.mean(current_points, axis=0)
        tgt_centroid = np.mean(previous_points, axis=0)

        # centering both point sets
        src_centered = current_points - src_centroid
        tgt_centered = previous_points - tgt_centroid

        H = src_centered.T @ tgt_centered  # form the cross-covariance
        U, S, V_T = np.linalg.svd(H)  # take SVD
        return (U, S, V_T)
        # rotation and translation done in perform_icp

    def publish_accumulated_cloud(self, stamp):
        """Publishes the existing accumulated pointcloud"""
        header = Header()
        header.stamp = stamp
        header.frame_id = "odom"

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, self.accumulated_points)
        self.pc_pub.publish(cloud_msg)
        self.get_logger().info("Published accumulated cloud.")

    def publish_icp_merged_cloud(self, stamp):
        """Publishes merged pointcloud"""
        header = Header()
        header.stamp = stamp
        header.frame_id = "odom"

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, self.icp_accumulated_points)
        self.icp_pub.publish(cloud_msg)
        self.get_logger().info("Published ICP merged cloud.")

def main(args=None):
    """Start ROS node"""
    rclpy.init(args=args)
    node = PauseAndCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()