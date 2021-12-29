curl -v POST http://127.0.0.1:5000/api/extract/0000-0000-0000-0000 \
  -d @./payloads/srcFile-regions-001.json \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer MY_API_KEY"
