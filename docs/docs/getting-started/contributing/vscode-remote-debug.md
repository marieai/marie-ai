# VSCode Remote Debug

Using VSCode Remote Debugging with Marie is easy.

## Starting the server

To start the server with `DEBUG` enabled, run the following command:

```shell
MARIE_DEBUG=1 marie server --start --uses /mnt/data/marie-ai/config/service/marie.yml 
```
This will start the server with `DEBUG` enabled and will wait for the debugger to attach.
By default, the debugger will wait for the client to attach on port `5678` and will bind to `0.0.0.0`

###  Advanced configuration:

* `MARIE_DEBUG` - enable debug
* `MARIE_DEBUG_PORT` - port to bind to
* `MARIE_DEBUG_WAIT_FOR_CLIENT` - wait for client to attach
* `MARIE_DEBUG_HOST` - host to bind to

If not specified, the following defaults will be used:

```shell
MARIE_DEBUG=1
MARIE_DEBUG_PORT=5678
MARIE_DEBUG_WAIT_FOR_CLIENT=1
MARIE_DEBUG_HOST=0.0.0.0
```

Usage:
```shell
MARIE_DEBUG=1;MARIE_DEBUG_PORT=5678;MARIE_DEBUG_WAIT_FOR_CLIENT=1;MARIE_DEBUG_HOST=0.0.0.0 \
marie server --start --uses /mnt/data/marie-ai/config/service/marie.yml 
```

## Attaching the debugger

To attach the debugger, you need to create or modify `launch.json` file in your `.vscode` folder.

Under the Run/Debug icon and then > menu `Run`, select `Add configuration` or `Run> Open Configurations`. 
Paste this in or add the keys, confirm your port matches the Marie session. https://code.visualstudio.com/docs/python/debugging

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
      // Attaches to injected debugpy hosted on port 5678.
      // https://code.visualstudio.com/docs/containers/docker-compose
      {
        "name": "Python: Remote Attach Marie (docker)",
        "type": "python",
        "request": "attach",
        "connect": {
          "host": "0.0.0.0",
          "port": 5678
        },
        "pathMappings": [
          {
            "localRoot": "${workspaceFolder}",
            "remoteRoot": "."
          }
        ]
      },
      // Attaches to injected debugpy hosted on port 5678.
      // https://code.visualstudio.com/docs/containers/docker-compose
      {
        "name": "Python: Remote Attach Marie (localhost)",
        "type": "python",
        "request": "attach",
        "connect": {
          "host": "0.0.0.0",
          "port": 5678
        },
        // "pathMappings": [
        //   {
        //     "localRoot": "${workspaceFolder}",
        //     "remoteRoot": "./marie"
        //   }
        // ]
      }
    ]
  }
```

## Debugging
At this point, you should be able to start the server and attach the debugger.
You can now set breakpoints and debug the server.

