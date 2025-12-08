import csv
import os
import subprocess
import math

def create_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows(data)

def main():
    # Create dummy lidar_odom.csv
    # t=0, x=0
    # t=100, x=100
    lo_data = [
        [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [100, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]
    create_csv('test_lo.csv', lo_data)

    # Create dummy vo.csv
    # t=50 (midpoint)
    vo_data = [
        [50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # Values here don't matter, only timestamp
    ]
    create_csv('test_vo.csv', vo_data)

    print("Running sync_odom.py...")
    subprocess.check_call(['python3', 'sync_odom.py', '--vo', 'test_vo.csv', '--lo', 'test_lo.csv', '--output', 'test_ts_lo.csv'])

    print("Checking output...")
    with open('test_ts_lo.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if len(rows) != 1:
        print("Error: Expected 1 row in output.")
        exit(1)
        
    row = rows[0]
    ts = int(row['timestamp'])
    tx = float(row['tx'])
    
    print(f"Timestamp: {ts}, tx: {tx}")
    
    if ts != 50:
        print("Error: Timestamp mismatch.")
        exit(1)
        
    if not math.isclose(tx, 50.0):
        print(f"Error: tx mismatch. Expected 50.0, got {tx}")
        exit(1)
        
    print("Verification passed!")
    
    # Clean up
    os.remove('test_lo.csv')
    os.remove('test_vo.csv')
    os.remove('test_ts_lo.csv')

if __name__ == "__main__":
    main()
