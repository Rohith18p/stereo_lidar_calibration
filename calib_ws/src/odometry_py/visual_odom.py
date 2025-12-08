import argparse
import csv
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from nav_msgs.msg import Odometry

def get_transform_matrix(tx, ty, tz, qx, qy, qz, qw):
    """
    Constructs a 4x4 transformation matrix from translation and quaternion.
    """
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    r = R.from_quat([qx, qy, qz, qw])
    T[:3, :3] = r.as_matrix()
    return T

def get_relative_transform(T_prev, T_curr):
    """
    Computes the relative transform T_rel such that T_prev * T_rel = T_curr.
    T_rel = T_prev_inv * T_curr
    """
    T_prev_inv = np.linalg.inv(T_prev)
    T_rel = np.dot(T_prev_inv, T_curr)
    return T_rel

def matrix_to_pose(T):
    """
    Extracts translation and quaternion from a 4x4 transformation matrix.
    """
    tx, ty, tz = T[:3, 3]
    r = R.from_matrix(T[:3, :3])
    qx, qy, qz, qw = r.as_quat()
    return tx, ty, tz, qx, qy, qz, qw

def main():
    parser = argparse.ArgumentParser(description="Extract relative visual odometry from ROS2 bag (mcap or db3).")
    parser.add_argument("--input_bag", default="/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/keyframes_bag_testtttt3333/keyframes_bag_testtttt3333_0.db3", help="Path to the input bag directory or file")
    parser.add_argument("--topic", default="/visual_odom", help="Odometry topic name")
    parser.add_argument("--output", default="visual_odom.csv", help="Output CSV file path")
    parser.add_argument("--storage_id", default="", help="Storage ID (e.g., mcap, sqlite3). If empty, tries to infer.")
    args = parser.parse_args()

    print(f"Reading {args.input_bag}...")
    print(f"Extracting topic: {args.topic}")
    print(f"Writing to: {args.output}")

    prev_T = None
    prev_timestamp = None
    
    # Initialize rosbag reader
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.input_bag, storage_id=args.storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        sys.exit(1)

    # Set filter for topic
    storage_filter = rosbag2_py.StorageFilter(topics=[args.topic])
    reader.set_filter(storage_filter)

    # Open CSV for writing
    with open(args.output, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Header: timestamp is the CURRENT timestamp (t2), transform is relative (t1->t2)
        csv_writer.writerow(['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        
        count = 0
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            
            if topic != args.topic:
                continue
                
            msg = deserialize_message(data, Odometry)
            
            # Handle timestamp
            if hasattr(msg, 'header'):
                sec = msg.header.stamp.sec
                nanosec = msg.header.stamp.nanosec
                timestamp = sec * 1e9 + nanosec
            else:
                timestamp = t

            # Extract pose
            try:
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                tx, ty, tz = pos.x, pos.y, pos.z
                qx, qy, qz, qw = ori.x, ori.y, ori.z, ori.w
            except AttributeError:
                print(f"Error: Message on topic {args.topic} does not have expected Odometry structure.")
                continue

            curr_T = get_transform_matrix(tx, ty, tz, qx, qy, qz, qw)

            if prev_T is not None:
                # Calculate relative transform from prev to curr
                T_rel = get_relative_transform(prev_T, curr_T)
                
                # Convert back to translation and quaternion
                rel_tx, rel_ty, rel_tz, rel_qx, rel_qy, rel_qz, rel_qw = matrix_to_pose(T_rel)
                
                # Write to CSV
                csv_writer.writerow([int(timestamp), rel_tx, rel_ty, rel_tz, rel_qx, rel_qy, rel_qz, rel_qw])
                count += 1

            prev_T = curr_T
            prev_timestamp = timestamp

    print(f"Done. Processed {count} relative poses.")

if __name__ == "__main__":
    main()
