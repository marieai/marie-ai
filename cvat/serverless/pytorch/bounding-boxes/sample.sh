#image=$(curl https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png --output - | base64 | tr -d '\n')
image=$(cat /home/greg/Downloads/yolo-sample/sample.png | base64 | tr -d '\n')
cat << EOF > /tmp/input.json
{"image": "$image"}
EOF
cat /tmp/input.json | nuctl invoke pth.marieai.bboxes -c 'application/json'
