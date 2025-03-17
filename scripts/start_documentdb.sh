# this is for testing only
# we will build a docker image for production use later
echo "Starting DocumentDB..."

docker run -d \
  --restart on-failure \
  --name marie-documentdb \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=123456 \
  -e POSTGRES_DB=postgres \
  -v ./data:/var/lib/postgresql/data \
  -p 5432:5432 \
  ghcr.io/ferretdb/postgres-documentdb:16