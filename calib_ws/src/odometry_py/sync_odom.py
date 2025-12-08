import argparse
import csv
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def read_csv(filepath):
    """
    Reads CSV file and returns a list of dictionaries.
    Assumes header: timestamp, tx, ty, tz, qx, qy, qz, qw
    """
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert values to float/int
                item = {
                    'timestamp': int(row['timestamp']),
                    'tx': float(row['tx']),
                    'ty': float(row['ty']),
                    'tz': float(row['tz']),
                    'qx': float(row['qx']),
                    'qy': float(row['qy']),
                    'qz': float(row['qz']),
                    'qw': float(row['qw']),
                }
                data.append(item)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)
    return data

def interpolate_pose(t, t1, pose1, t2, pose2):
    """
    Interpolates between pose1 at t1 and pose2 at t2 for time t.
    """
    if t1 == t2:
        return pose1

    alpha = (t - t1) / (t2 - t1)

    # Linear interpolation for position
    tx = (1 - alpha) * pose1['tx'] + alpha * pose2['tx']
    ty = (1 - alpha) * pose1['ty'] + alpha * pose2['ty']
    tz = (1 - alpha) * pose1['tz'] + alpha * pose2['tz']

    # SLERP for orientation
    # Create Rotation objects
    r1 = R.from_quat([pose1['qx'], pose1['qy'], pose1['qz'], pose1['qw']])
    r2 = R.from_quat([pose2['qx'], pose2['qy'], pose2['qz'], pose2['qw']])
    
    # Scipy Slerp requires a times array and a Rotation object containing the rotations
    key_times = [t1, t2]
    key_rots = R.from_quat([
        [pose1['qx'], pose1['qy'], pose1['qz'], pose1['qw']],
        [pose2['qx'], pose2['qy'], pose2['qz'], pose2['qw']]
    ])
    
    slerp = Slerp(key_times, key_rots)
    r_interp = slerp([t])
    qx, qy, qz, qw = r_interp.as_quat()[0]

    return {
        'timestamp': int(t),
        'tx': tx, 'ty': ty, 'tz': tz,
        'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw
    }

def main():
    parser = argparse.ArgumentParser(description="Synchronize lidar odometry to visual odometry timestamps.")
    parser.add_argument("--vo", default="vo.csv", help="Path to visual odometry CSV (target timestamps)")
    parser.add_argument("--lo", default="lidar_odom.csv", help="Path to lidar odometry CSV (source poses)")
    parser.add_argument("--output", default="ts_lo.csv", help="Output CSV file path")
    args = parser.parse_args()

    print(f"Reading {args.vo}...")
    vo_data = read_csv(args.vo)
    
    print(f"Reading {args.lo}...")
    lo_data = read_csv(args.lo)

    # Sort lo_data by timestamp just in case
    lo_data.sort(key=lambda x: x['timestamp'])
    
    lo_timestamps = [x['timestamp'] for x in lo_data]
    
    interpolated_data = []
    
    print("Synchronizing...")
    
    # Optimize search by keeping track of last index
    last_idx = 0
    
    for vo_row in vo_data:
        t_target = vo_row['timestamp']
        
        # Find range [t_prev, t_next] in lo_data
        # We want lo_data[i].timestamp <= t_target <= lo_data[i+1].timestamp
        
        # Advance last_idx until we find the window or run out
        while last_idx < len(lo_data) - 1 and lo_data[last_idx+1]['timestamp'] < t_target:
            last_idx += 1
            
        if last_idx >= len(lo_data) - 1:
            # t_target is after the last lidar timestamp
            # Check if it's exactly the last one
            if lo_data[-1]['timestamp'] == t_target:
                 interpolated_data.append(lo_data[-1])
            else:
                # Cannot interpolate
                # print(f"Warning: Timestamp {t_target} is out of range (after end). Skipping.")
                pass
            continue
            
        if lo_data[last_idx]['timestamp'] > t_target:
             # t_target is before the first lidar timestamp (shouldn't happen if we start at 0, but possible)
             # print(f"Warning: Timestamp {t_target} is out of range (before start). Skipping.")
             continue
             
        # Now we have lo_data[last_idx] <= t_target <= lo_data[last_idx+1]
        t1 = lo_data[last_idx]['timestamp']
        t2 = lo_data[last_idx+1]['timestamp']
        
        pose1 = lo_data[last_idx]
        pose2 = lo_data[last_idx+1]
        
        interp_pose = interpolate_pose(t_target, t1, pose1, t2, pose2)
        interpolated_data.append(interp_pose)

    print(f"Writing {len(interpolated_data)} synchronized poses to {args.output}...")
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        writer.writeheader()
        writer.writerows(interpolated_data)
        
    print("Done.")

if __name__ == "__main__":
    main()
