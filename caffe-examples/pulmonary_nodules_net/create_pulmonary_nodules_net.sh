#!/usr/bin/env sh
# Create the pulmonary_nodules_net lmdb inputs
# N.B. set the path to the pulmonary_nodules_net train + val data dirs
set -e

EXAMPLE=examples/pulmonary_nodules_net
DATA=data/pulmonary_nodules
TOOLS=build/tools

TRAIN_DATA_ROOT=../Pulmonary_nodules_data/train/
VAL_DATA_ROOT=../Pulmonary_nodules_data/val/

# Set RESIZE=true to resize the images to 512x512. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=512
  RESIZE_WIDTH=512
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_pulmonary_nodules_net.sh to the path" \
       "where the pulmonary_nodules_net training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_pulmonary_nodules_net.sh to the path" \
       "where the pulmonary_nodules_net validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/pulmonary_nodules_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/pulmonary_nodules_val_lmdb

echo "Done."
