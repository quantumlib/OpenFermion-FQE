#!/bin/bash

cd ..
sudo python3 setup.py install
cd docs
make html
