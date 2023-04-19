#image=$(curl https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png --output - | base64 | tr -d '\n')
image=$(cat /home/gbugaj/Downloads/yolo/sample.jpg | base64 | tr -d '\n')
cat << EOF > /tmp/input.json
{"image": "$image"}
EOF
cat /tmp/input.json | nuctl invoke openvino-omz-public-yolo-v3-tf -c 'application/json'
