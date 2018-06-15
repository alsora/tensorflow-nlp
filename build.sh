#!/bin/bash
#
# Musixmatch Intelligence SDK Docker Build Script
# @author Alberto Soragna (alberto dot soragna at musixmatch dot com)
# @2018 Musixmatch
    
echo Building $MXM_BASE_CONTAINER:$MXM_VERSION...

#[ -z "$AWS_ACCESS_KEY_ID" ] && { echo "Missing AWS_ACCESS_KEY_ID. Please run first"; echo "source env.sh"; exit 1; }
#[ -z "$AWS_SECRET_ACCESS_KEY" ] && { echo "Missing AWS_SECRET_ACCESS_KEY. Please run first"; echo "source env.sh"; exit 1; }
#docker build -t $MXM_BASE_CONTAINER:$MXM_VERSION --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY .

docker build -t $MXM_BASE_CONTAINER:$MXM_VERSION .