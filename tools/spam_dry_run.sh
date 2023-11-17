set -B                  # enable brace expansion
for i in {1..2}; do
#   curl -s -k 'GET' -H 'header info' -b 'stuff' 'http://192.168.102.53:51000/dry_run?id='$i
  curl -s -k 'GET' 'http://172.20.10.49:51000/dry_run'
done
