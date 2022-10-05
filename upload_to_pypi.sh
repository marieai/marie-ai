#!/bin/bash

git pull origin master

old_version=$(grep -Po "(?<=version=')[^']+(?=')" setup.py)
echo "Current version is $old_version. New version?"
read new_version
sed -i "s/version='$old_version'/version='$new_version'/g" setup.py