.PHONY: build test run

build:
	go build -o dist/marie-plugin-daemon ./cmd/server

test:
	go test ./...

run:
	go run ./cmd/server --addr 127.0.0.1:8099
