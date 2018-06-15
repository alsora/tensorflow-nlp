#!/bin/bash
#
# Musixmatch Intelligence SDK Docker Build Script
# @author Alberto Soragna (alberto dot soragna at musixmatch dot com)
# @2018 Musixmatch
    
echo Building $MXM_BASE_CONTAINER:$MXM_VERSION...

docker build -t $MXM_BASE_CONTAINER:$MXM_VERSION .