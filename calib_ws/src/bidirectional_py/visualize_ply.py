import open3d as o3d
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize PLY Point Cloud")
    parser.add_argument("file", nargs='?', default="debug_stereo_notebook.ply", help="Path to PLY file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    print(f"Loading {args.file}...")
    pcd = o3d.io.read_point_cloud(args.file)
    
    print("Visualizing... (Close the window to exit)")
    
    # Add a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, coord_frame], 
                                      window_name=f"Point Cloud Viewer - {args.file}",
                                      width=1024, height=768)

if __name__ == "__main__":
    main()
