#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/3D-CV-Project/data/car90/"
DEST_DIR="/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/3D-CV-Project/data/car90/gt_camera_params"
DEST2_DIR="/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/3D-CV-Project/data/car90/images"

if [ ! -d "$SOURCE_DIR" ]; then
  mkdir "$SOURCE_DIR"
else
  echo "Directory $DIRECTORY already exists."
fi

if [ ! -d "$DEST_DIR" ]; then
  mkdir "$DEST_DIR"
else
  echo "Directory $DEST_DIR already exists."
fi

if [ ! -d "$DEST2_DIR" ]; then
  mkdir "$DEST2_DIR"
else
  echo "Directory $DEST2_DIR already exists."
fi

# Move .npy and .pkl files from the source to the destination directory
mv "$SOURCE_DIR"/*.npy "$DEST_DIR"
mv "$SOURCE_DIR"/*.pkl "$DEST_DIR"
mv "$SOURCE_DIR"/*.png "$DEST2_DIR"

# Print a message indicating completion
echo "Moved all .npy and .pkl files from $SOURCE_DIR to $DEST_DIR"
