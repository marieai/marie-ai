MARIE_DEFAULT_MOUNT ?= /mnt/data/marie-ai
BINDIR ?= $(MARIE_DEFAULT_MOUNT)/bin

.PHONY: build install test run

build:
	go build -o dist/marie-plugin-daemon ./cmd/server

install:
	mkdir -p "$(BINDIR)"
	go build -o "$(BINDIR)/marie-plugin-daemon" ./cmd/server

test:
	go test ./...

run: install
	"$(BINDIR)/marie-plugin-daemon"
