#!/bin/bash
#
# Musixmatch Intelligence SDK Docker Build Script
# @author Alberto Soragna (alberto dot soragna at musixmatch dot com)
# @2018 Musixmatch

[ -z "$MXM_BASE_NAME" ] && { echo "Missing environment. Please run first"; echo "source env.sh"; exit 1; }
MXM_VERSION=$1

[ -z "$MXM_VERSION" ] && { echo "Usage: $0 VERSION"; exit 1; }

ECHO Using...
ECHO Keys path:$KEYS_PATH \$KEYS_PATH
ECHO Volume path:$MXM_VOL_CONTAINER_PATH \$MXM_VOL_CONTAINER_PATH
ECHO Volume container:$MXM_VOL_CONTAINER \$MXM_VOL_CONTAINER
ECHO Volume image:$MXM_VOL_NAME \$MXM_VOL_NAME
ECHO Base container:$MXM_BASE_CONTAINER \$MXM_BASE_CONTAINER
ECHO Base image:$MXM_BASE_NAME \$MXM_BASE_NAME
ECHO Version: $MXM_VERSION



CMD=bash

## Simple run
docker run -it --name $MXM_BASE_NAME $MXM_BASE_CONTAINER:$MXM_VERSION $CMD

