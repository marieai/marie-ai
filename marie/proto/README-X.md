# Build proto

Jina support two versions of protobuf, before 3.19 and after (which is a breaking change for python), therefore we have
duplicate python file generation from proto based on the installed protobuf version.

Moreover, jina is compatible with docarray v1 and docarray v2 that introduce breaking change in the proto definition.

Therefore, we end with 2 proto files, one for each version of docarray that we support. (Note in the future 
docarray v1 support will be dropped, and we will come back to have proto version)

This complex setup lead to a not straightforward way to generate the python code from the proto

this guide explain how to do it properly.

# how to build the proto

## 1. build docker image for protobuf generation


``` cmd
docker build -f Dockerfiles/protogen.Dockerfile -t marieai/protogen:local .
```

This build the docker image that will be used to generate the python code from the proto for proto **before** 3.19

``` cmd
docker build -f Dockerfiles/protogen-3.21.Dockerfile -t marieai/protogen-3.21:local .
```

This build the docker image that will be used to generate the pytfixhon code from the proto for proto **after* 3.19

## 2. generate the python code from the proto

note: you need to be in the root of the repo to do the following steps

### For DocArray v1

``` cmd
docker run -it -v $(pwd)/marie/proto/docarray_v1:/marie/proto marieai/protogen:local
```

``` cmd
docker run -it -v $(pwd)/marie/proto/docarray_v1:/marie/proto marieai/protogen-3.21:local
```

### For DocArray v2

``` cmd
docker run -it -v $(pwd)/marie/proto/docarray_v2:/marie/proto marieai/protogen:local
```

``` cmd
docker run -it -v $(pwd)/marie/proto/docarray_v2:/marie/proto marieai/protogen-3.21:local
```

