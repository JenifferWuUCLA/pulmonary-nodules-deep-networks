#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/pulmonary_nodules_net_caffenet/solver.prototxt \
    --snapshot=models/pulmonary_nodules_net_caffenet/caffenet_train_10000.solverstate.h5 \
    $@
