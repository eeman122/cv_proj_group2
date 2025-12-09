#!/usr/bin/env python3
"""
Quick-start script to run SfM on your personal images.
This script sets up and runs the complete structure-from-motion pipeline.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Run SfM pipeline on your personal images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all images with SIFT features
  python run_sfm.py
  
  # Use first 20 images with SURF features
  python run_sfm.py --num_images 20 --feat_type surf
  
  # Custom paths
  python run_sfm.py --source_dir /path/to/images --root_dir /path/to/sfm
        """
    )
    
    parser.add_argument(
        '--source_dir',
        type=str,
        default='/Users/eemanadnan/Documents/MS-AI/Assignments/3D_Scene_Reconstruction/images_folder',
        help='Path to your images folder'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='/Users/eemanadnan/Documents/MS-AI/Assignments/structure-from-motion-master',
        help='Root directory for SfM'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=None,
        help='Limit number of images (default: all)'
    )
    parser.add_argument(
        '--feat_type',
        type=str,
        default='sift',
        choices=['sift', 'surf', 'orb'],
        help='Feature type (default: sift)'
    )
    parser.add_argument(
        '--skip_setup',
        action='store_true',
        help='Skip setup and go straight to SfM'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Structure from Motion - Personal Images Pipeline")
    print("=" * 70)
    
    # Change to SfM directory
    os.chdir(args.root_dir)
    print(f"\nWorking directory: {os.getcwd()}\n")
    
    # Step 1: Setup (copy images and create calibration)
    if not args.skip_setup:
        print("Step 1: Setting up images and camera calibration...")
        print("-" * 70)
        setup_cmd = [
            sys.executable, 'setup_for_user_images.py',
            '--source_dir', args.source_dir,
            '--root_dir', args.root_dir
        ]
        if args.num_images:
            setup_cmd.extend(['--num_images', str(args.num_images)])
        
        result = subprocess.run(setup_cmd, cwd=args.root_dir)
        if result.returncode != 0:
            print("\n✗ Setup failed!")
            sys.exit(1)
    
    # Step 2: Run SfM
    print("\n" + "=" * 70)
    print("Step 2: Running SfM reconstruction...")
    print("=" * 70 + "\n")
    
    sfm_cmd = [
        sys.executable, 'main.py',
        '--root_dir', args.root_dir,
        '--feat_type', args.feat_type,
        '--image_format', 'jpg'
    ]
    
    result = subprocess.run(sfm_cmd, cwd=args.root_dir)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✓ SfM reconstruction complete!")
        print("=" * 70)
        print(f"\nResults saved in: {os.path.join(args.root_dir, 'points')}/")
        print("\nTo visualize the 3D reconstruction:")
        print("  1. Open the .ply files in MeshLab")
        print("  2. Or convert them and view with other 3D software")
    else:
        print("\n✗ SfM reconstruction failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
