# Marie-AI Documentation


## Contribute to Documentation
Please have a look at our [contribution guide](https://github.com/marieai/marie-ai/blob/main/docs/docs/getting-started/contributing/contributing.md) to see how to install the development environment and how to generate the documentation.


### Installation

```
$ nvm install $(cat .nvmrc)
$ nvm use
$ npm install --location=global yarn
```

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

Using SSH:

```
$ USE_SSH=true yarn deploy
```

Not using SSH:

```
$ GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
