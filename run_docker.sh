#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

docker run -it \
  --rm --gpus all \
  -v ${SCRIPT_DIR}:/CamRaDepth \
  -v ${SCRIPT_DIR}/../nuscenes_mini:/nuscenes_mini \
  camradepth:latest
