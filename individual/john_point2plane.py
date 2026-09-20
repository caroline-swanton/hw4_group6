"""

This starter code provides code base to implement point to plane ICP for HW4 Q3
This is a stand alone code and does not require ROS
--- TBD --- indicates the section to fill out   
... indicates the specific place to type the code

NOTE: In the main function, there is a parameter skip_pose, where the program will skip
every skip_pose number of poses to speed up the process. You can set it to 1 to use all
the poses, but it will take longer to run. Ideally we recommend ~25 for initial testing 
but want it at 5 for the final results

"""

import re
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

class DataParser:
    """Parses odometry and scan data from log files.
    This class provides functionality to read and parse ROS2 bag file2txt dumps
    NO ***** TBD ********* items in this class
    """

    def __init__(self, odom_path, scan_path):
        self.odom_path = odom_path
        self.scan_path = scan_path
        self.odom_data = []
        self.scan_data = []
        self.x = []
        self.y = []
        self.theta = []

    def quaternion_to_yaw(self, z, w):
        """Calculate yaw from quat"""
        return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)

    # pylint: disable=too-many-locals
    def parse_odom(self):
        """Parses odometry"""
        results = []
        x = []
        y = []
        yaw = []
        timestamp = []
        # pylint: disable=unspecified-encoding
        with open(self.odom_path, 'r') as f:
            timestamp_ = None
            for line in f:
                line = line.strip()
                if line.startswith("# Time:"):
                    timestamp_ = float(line.split(":")[1].strip())
                elif line.startswith("nav_msgs.msg.Odometry("):
                    pos_match = re.search(
                        r"position=geometry_msgs\.msg\.Point\(x=([-\de.E]+), y=([-\de.E]+)", line
                        )
                    ori_match = re.search(
                        r"orientation=geometry_msgs\.msg\.Quaternion\(x=[-\de.E]+, y=[-\de.E]+, z=([-\de.E]+), w=([-\de.E]+)",
                        line
                        )
                    if pos_match and ori_match and timestamp_ is not None:
                        x_ = float(pos_match.group(1))
                        y_ = float(pos_match.group(2))
                        z_ = float(ori_match.group(1))
                        w_ = float(ori_match.group(2))
                        yaw_ = self.quaternion_to_yaw(z_, w_)
                        results.append((timestamp_, x_, y_, yaw_))
                        x.append(x_)
                        y.append(y_)
                        yaw.append(yaw_)
                        timestamp.append(timestamp_)
                        timestamp_ = None
        self.odom_data = np.vstack(results)
        self.x = np.array(x)
        self.y = np.array(y)
        self.theta = np.array(yaw)
        print("No. of odom parsed: ", self.theta.shape)

    # pylint: disable=too-many-locals
    def parse_scan(self):
        """Parses each scan"""
        # pylint: disable=unspecified-encoding
        with open(self.scan_path, 'r') as f:
            text = f.read()
        entries = text.split("# Time:")
        angle_min = []
        angle_max = []
        angle_increment = []
        ranges = []
        for entry in entries[1:]:
            time_match = re.search(r'(\d+\.\d+)', entry)
            scan_match = re.search(
                    r'angle_min=([\-\d.e]+), angle_max=([\-\d.e]+), '
                    r'angle_increment=([\-\d.e]+),.*?ranges=\[(.*?)\]',
                    entry,
                    re.DOTALL
                    )
            if time_match and scan_match:
                angle_min_ = float(scan_match.group(1))
                angle_max_ = float(scan_match.group(2))
                angle_increment_ = float(scan_match.group(3))
                ranges_str = scan_match.group(4).replace('\n', '').replace(' ', '')
                ranges_ = [float(r) if r != 'nan' else np.nan for r in ranges_str.split(',') if r]
                angle_min.append(angle_min_)
                angle_max.append(angle_max_)
                angle_increment.append(angle_increment_)
                ranges.append(ranges_)
        print("No. of lidar scan parsed: ", np.asarray(angle_increment).shape)
        self.scan_data = (ranges, angle_min, angle_max, angle_increment)


class Point2PlaneICP:
    """ICP calculation"""
    # pylint: disable=redefined-outer-name
    def __init__(self, parser, skip_pose=10):
        self.parser = parser
        self.skip_pose = skip_pose
        self.accumulated_points = []
        # set this to true to visualize the normals
        self.visualize_normal = False
        # use this to switch between a simple and more robust normal estimation
        self.normal_simple = True

    def ranges_to_xy(self, ranges, angles):
        """
        Polar coordinate to cartesian conversion. Borrowed from Hw4
        """
        ranges = np.array(ranges)
        angles = angles[:len(ranges)]
        valid = ~np.isnan(ranges)
        return np.vstack([
            ranges[valid] * np.cos(angles[valid]),
            ranges[valid] * np.sin(angles[valid])
        ]).T

    def transform_points(self, points, pose):
        """
        Odomtery based point cloud merge. here used for good initialization. Borrowed from HW4
        """
        x, y, theta = pose
        # pylint: disable=invalid-name
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        t = np.array([x, y])
        return (R @ points.T).T + t

    def points_to_normal(self, prev_point, next_point):
        """Helper function to calculate normal using central difference"""
        tangent = next_point - prev_point
        normal = [-tangent[1], tangent[0]]
        normal = normal / np.linalg.norm(normal)
        return normal

    def align_normal(self, point, normal, sensor_origin):
        """Helper function to align the normal towards the origin"""
        to_sensor = sensor_origin - point
        if np.dot(normal, to_sensor) < 0:
            normal = -normal
        return normal

    def compute_normals(self, points, sensor_origin):

        # --------------- TBD ---------------
        #************************************************
        #           IMPORTANT DETAILS BELOW             #
        #************************************************
        """
        Three point normals. i.e compute normals using 3 neighbors
        This is the simplest form of normals.
        A more sophisticated version is available under compute_normals_pca()
        using Principle Component Analysis
            - Use this function to check if your normals looks similar
            - To use compute_normals_pca(), set self.normal_simple = False

        - for point cloud index from 0 to n,
        - step 1: start from index 1 to n-1, i.e leaving the first and last point
        - step 2: in point clouds, normals = previous point - next point
            - this gives you a central difference approximation of the tangent
            - then rotate 90 deg to get the normal
            - e.g. tangent = p[i + 1] - p[i - 1] = [4, 5] - [2, 3] = [2, 2]
            - step 3: flip this tangent to get the normal. i.e. normal
              = [-tangent[1], tangent[0]] = [-2, 2]
        - step 4: find the normal = normal / norm(normal)
        - step 5: the sensor_origin is used to align normals in the same directions
        - step 6: append to the list normals[]
        - step 7: normals[0] = normals[1], and normals[n] = normals[n-1]
        - visualize the normals using the function plot_normals()
            i.e enable self.visualize_normal under init()
         """
        normals = [] # initialize
        n = len(points)
        # handle normals[0]
        norm_candidate_0 = self.points_to_normal(points[0], points[2])
        norm_true_0 = self.align_normal(points[1], norm_candidate_0, sensor_origin)
        normals.append(norm_true_0)
        for i in range(1, n-1):
            norm_candidate = self.points_to_normal(points[i - 1], points[i + 1])
            norm_true = self.align_normal(points[i], norm_candidate, sensor_origin)
            normals.append(norm_true)
        # handle normals[n]
        norm_candidate_n = self.points_to_normal(points[n - 3], points[n-1])
        norm_true_n = self.align_normal(points[n - 1], norm_candidate_n, sensor_origin)
        normals.append(norm_true_n)
        # visualize
        if self.visualize_normal:
            self.plot_normals(points, normals)
        return np.array(normals)

    def compute_normals_pca(self, points, tree, sensor_origin, k=10):
        """
        More sophisticated way to compute normals. Here using PCA
        This was not explicitly covered in class but similar to the concept
        sof SVD principal directions
        Use this function to visualize normals and compare it to your method
        """
        normals = []
        for pt in points:
            _, idxs = tree.query(pt, k=k)
            neighbors = points[idxs]
            cov = np.cov(neighbors.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            normal /= np.linalg.norm(normal)
            to_sensor = sensor_origin - pt
            if np.dot(normal, to_sensor) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)

    # pylint: disable=too-many-locals
    def icp_point_to_plane(self, source_points, target_points, \
                           target_sensor_origin, max_iterations=15):

        """
        This is the main function that computes the point to plane ICP

        """
        src = np.copy(source_points)
        tgt = np.copy(target_points)
        tree = cKDTree(tgt)

        # a mechanism to switch between your normals and the robust normals estimates
        if self.normal_simple:
            normals = self.compute_normals(tgt, target_sensor_origin)
        else:
            normals = self.compute_normals_pca(tgt, tree, target_sensor_origin, k=8)

        if self.visualize_normal:
            self.plot_normals(tgt, normals)


        # ----------------------- TBD -------------------
        # pylint: disable=unpacking-non-sequence
        for i in range(max_iterations):
            distances, indices = tree.query(src)  # use kd tree for initial association.
            matched_pts = tgt[indices]    # get the matched points from target PCL using the kd tree index
            matched_normals = normals[indices]  # get the matched normals using the kd tree index
            # pylint: disable=invalid-name
            A, b = [], [] # initialize
            for p, q, n in zip(src, matched_pts, matched_normals):
                # using δ, n (normals) form the parts of the linear system
                # i.e. A and b as the system is Ax=b
                # refer to the lecture slides for more details on the equations
                # A = [n Rp n] is the matrix form, use the linearized form for the Rp
                #   i.e. A = [n · [-py, px], nx, ny]
                # b = [-n δ] =>  b = -n · (p - q)
                # when working with vectors, use dot product where applicable
                # Append A and b to the list

                # build A and b
                residual = p - q
                b_i = np.dot(-n, residual)
                b.append(b_i)
                nx = n[0]
                ny = n[1]
                px = p[0]
                py = p[1]
                a_i = [np.dot(n, [-py, px]), nx, ny]
                A.append(a_i)

            # pylint: disable=invalid-name
            A = np.array(A)
            b = np.array(b)

            # use the np.linalg.lstsq() to solve for the least square.
            # This function performs SVD in the background but is more stable to conventional SVD.
            # extract the angle, tx, ty from the above step
            # form a rotation matrix, and a translation vector
            # transform the source using the R and t from above
            # alternatively form a homogeneous transformation matrix and transform the source
            # return source point cloud

            # get rotation and translation
            x, lsq_residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            theta = x[0]
            t_x = x[1]
            t_y = x[2]
            translation_vector = np.array([t_x, t_y])
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            # apply transformation
            # src is Nx2, R is 2x2, t is 1x2
            src = (rotation_matrix @ src.T).T + translation_vector

            if abs(theta) < 0.00001 and np.linalg.norm(translation_vector) < 0.00001:
                print(f"number of iterations: {i + 1}")
                print(f"final mean absolute residual: {np.mean(np.abs(b))}")
                break

            # ----------------------- TBD-END -------------------
        return src

    def plot_normals(self, points, normals, title="Point Normals"):
        """
        plot the normals
        """
        points = np.array(points)
        normals = np.array(normals)
        plt.figure(figsize=(8, 8))
        plt.quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1],
                   angles='xy', scale_units='xy', scale=15, color='r', width=0.003, label='Normals')
        plt.plot(points[:, 0], points[:, 1], 'b.', markersize=2, label='Points')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def visualize_odom(self):
        """
        diplay's the robot's movement while collecting the laser scans
        """
        x, y, theta = self.parser.x, self.parser.y, self.parser.theta
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker='o', markersize=2, linestyle='-', label="Odometry Path")
        dx = np.cos(theta)
        dy = np.sin(theta)
        plt.quiver(x[::10], y[::10], dx[::10], dy[::10], angles='xy',
                   scale_units='xy', scale=15, color='r',
                   width=0.005, label='Orientation')
        plt.title("Odometry Trajectory with Orientation")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

    # pylint: disable=too-many-locals
    def run_icp(self):
        """
        This is the function that reads the data and runs icp in loop

        """
        accumulated_pc = None
        ranges, angle_mins, angle_maxs, _ = self.parser.scan_data
        x = self.parser.x
        y = self.parser.y
        theta = self.parser.theta
        for i in range(0, len(ranges), self.skip_pose):
            r = ranges[i]
            angles = np.linspace(angle_mins[i], angle_maxs[i], len(r))
            local_pts = self.ranges_to_xy(r, angles)
            pose = (x[i], y[i], theta[i])
            # this step makes it easier for icp to converge
            initialized_pts = self.transform_points(local_pts, pose)
            sensor_origin = np.array([pose[0], pose[1]])
            if accumulated_pc is None:
                accumulated_pc = initialized_pts
                self.accumulated_points.append(initialized_pts)
            else:
                aligned = self.icp_point_to_plane(initialized_pts, accumulated_pc, sensor_origin)
                accumulated_pc = np.vstack((accumulated_pc, aligned))
                self.accumulated_points.append(aligned)
            print(f"Progress: {np.round(i / len(ranges) * 100.0, 2)}%", end='\r')

    def plot_map(self):
        """Plots map"""
        plt.figure(figsize=(8, 8))
        for pc in self.accumulated_points:
            plt.scatter(pc[:, 0], pc[:, 1], s=1)
        plt.title("Accumulated Point Cloud via Point-to-Plane ICP")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    path_ = "data/" # change this to match your directory. note: '/' expected at the end
    parser = DataParser(odom_path=path_ + "odom_sync.txt", scan_path=path_ + "scan_sync.txt")
    parser.parse_odom()
    parser.parse_scan()
    mapper = Point2PlaneICP(parser=parser, skip_pose=5)
    mapper.visualize_odom()
    mapper.run_icp()
    mapper.plot_map()
