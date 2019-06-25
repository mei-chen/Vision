#!/bin/sh

set -ex

docker build -t zihao/play:visn-app -f deploy/app.dockerfile .
docker build -t zihao/play:visn-www -f deploy/www.dockerfile .
docker push zihao/play:visn-app
docker push zihao/play:visn-www
kubectl apply -f deploy/
kubepatch visn-app
kubepatch visn-www
