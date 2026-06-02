FROM golang:1.25-alpine AS build

WORKDIR /src
COPY . .
RUN go build -o /out/marie-plugin-daemon ./cmd/server

FROM alpine:3.22

COPY --from=build /out/marie-plugin-daemon /usr/local/bin/marie-plugin-daemon
EXPOSE 8099
ENTRYPOINT ["marie-plugin-daemon", "--addr", "0.0.0.0:8099"]
