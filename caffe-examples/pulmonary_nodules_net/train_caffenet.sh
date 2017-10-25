#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/pulmonary_nodules_net_caffenet/solver.prototxt $@
