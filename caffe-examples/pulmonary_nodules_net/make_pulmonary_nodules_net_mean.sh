#!/usr/bin/env sh
# Compute the mean image from the pulmonary_nodules_net training lmdb
# N.B. this is available in data/pulmonary_nodules

EXAMPLE=examples/pulmonary_nodules_net
DATA=data/pulmonary_nodules
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/pulmonary_nodules_train_lmdb \
  $DATA/pulmonary_nodules_net_mean.binaryproto

echo "Done."
