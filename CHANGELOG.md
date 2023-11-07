
















<a name=release-note-__version__
=

"3-0-0"></a>

## Release Note (`__version__ = "3.0.0"`)


> Release time: 2023-03-23 13:35:35



🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-__version__
=

"3-0-0"></a>

## Release Note (`__version__ = "3.0.0"`)


> Release time: 2023-03-23 13:42:52



🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-__version__
=

"3-0-0"

__version__


=

"3-0-01></a>
## Release Note (`__version__ = "3.0.0"
__version__ = "3.0.01`)

> Release time: 2023-03-23 13:45:04



🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-__version__
=

"3-0-02"

__version__


=

"3-0-0"
__version__

=

"3-0-02></a>


## Release Note (`__version__ = "3.0.02"
__version__ = "3.0.0" __version__ = "3.0.02`)

> Release time: 2023-03-23 13:45:29


🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-__version__
=

"3-0-03"

__version__


=

"3-0-02"
__version__

=

"3-0-0"


__version__

=
"3-0-03></a>

## Release Note (`__version__ = "3.0.03"
__version__ = "3.0.02" __version__ = "3.0.0" __version__ = "3.0.03`)

> Release time: 2023-03-23 13:56:57




🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-__version__
=

"3-0-03"

__version__


=

"3-0-02"
__version__

=

"3-0-0"


__version__

=
"3-0-04

__version__

=


"3-0-03"

__version__
=

"3-0-02"

__version__


=

"3-0-0"
__version__

=

"3-0-04></a>


## Release Note (`__version__ = "3.0.03" __version__ = "3.0.02" __version__ = "3.0.0" __version__ = "3.0.04
__version__ = "3.0.03" __version__ = "3.0.02" __version__ = "3.0.0" __version__ = "3.0.04`)

> Release time: 2023-03-23 13:58:00


🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-__version__
=

"3-0-06"

__version__


=

"3-0-03"
__version__

=

"3-0-02"


__version__

=
"3-0-0"

__version__

=


"3-0-04

__version__
=

"3-0-03"

__version__


=

"3-0-02"
__version__

=

"3-0-0"


__version__

=
"3-0-05></a>

## Release Note (`__version__ = "3.0.06"
__version__ = "3.0.03" __version__ = "3.0.02" __version__ = "3.0.0" __version__ = "3.0.04 __version__ = "3.0.03" __version__ = "3.0.02" __version__ = "3.0.0" __version__ = "3.0.05`)

> Release time: 2023-03-23 13:59:38




🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-__version__
=

"3-0-0"></a>

## Release Note (`__version__ = "3.0.0"`)


> Release time: 2023-03-23 14:01:31



🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-__version__
=

"3-0-01"></a>

## Release Note (`__version__ = "3.0.01"`)


> Release time: 2023-03-23 14:02:22



🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-3-0-01></a>
## Release Note (`3.0.01`)

> Release time: 2023-03-23 14:02:37



🙇 We'd like to thank all contributors for this new release! In particular,
 gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)
 - [[```734aac12```](https://github.com/marieai/marie-ai/commit/734aac12b8d4c6a7e72a091760fedd08e115a489)] __-__ initial test_interface tests are passing (*gbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)
 - [[```756e1071```](https://github.com/marieai/marie-ai/commit/756e1071260b76f2afabaf03f3046e74b53b7f3f)] __-__ __ci__: github workflow integration (*Greg*)
 - [[```b1ca2f5a```](https://github.com/marieai/marie-ai/commit/b1ca2f5a2b7df10894d56b5f994f6b366607cc34)] __-__ add hello-marie example (*gbugaj*)
 - [[```74f0b1fc```](https://github.com/marieai/marie-ai/commit/74f0b1fc24f2c4dc1f084df42ac1fcdd4c5a65a1)] __-__ latest jina merge (*gbugaj*)
 - [[```f40efc67```](https://github.com/marieai/marie-ai/commit/f40efc67cadf8ccc76b1aa093f60bdae5c137f3f)] __-__ itegrating changes from Jina main (*gbugaj*)
 - [[```ade5cf4b```](https://github.com/marieai/marie-ai/commit/ade5cf4b2b0f51761dd41464a4c5b6dbc7c02d20)] __-__ jina integration (*Greg*)
 - [[```0b491a88```](https://github.com/marieai/marie-ai/commit/0b491a880a45b6de0be5ad652eb9a175f4ea8598)] __-__ integrating JINA tests (*gbugaj*)
 - [[```b0e3ebb8```](https://github.com/marieai/marie-ai/commit/b0e3ebb834f4e33c08ecac93b5da3a9f6eedab50)] __-__ mrege latest jina (*gbugaj*)
 - [[```e14fff59```](https://github.com/marieai/marie-ai/commit/e14fff5918b453ad91ecf59daecb42aae9656571)] __-__ merged latest jina parser and schema (*gbugaj*)
 - [[```8f7744ae```](https://github.com/marieai/marie-ai/commit/8f7744ae4718743e1c4ced1394135b9125f72a55)] __-__ integration of jina jaml (*gbugaj*)
 - [[```3a33fb41```](https://github.com/marieai/marie-ai/commit/3a33fb41572b2246e0984d696b4760c849cf00f7)] __-__ integrated latest jina client (*gbugaj*)
 - [[```1488998e```](https://github.com/marieai/marie-ai/commit/1488998e3e1f67c227d072556e97cc5a6bb84dd3)] __-__ integrated head from jina (*Greg*)
 - [[```39f2bf91```](https://github.com/marieai/marie-ai/commit/39f2bf91ddef11b60caf1af3d0e6b5c6c732ce1c)] __-__ integrated orchestrate from jina (*Greg*)
 - [[```47aa5466```](https://github.com/marieai/marie-ai/commit/47aa5466a6357f3c16a1941da2e8ab0fe30d61b3)] __-__ integrated serve runtimes from jina (*Greg*)

<a name=release-note-3-0-02></a>
## Release Note (`3.0.02`)

> Release time: 2023-03-23 15:15:34



🙇 We'd like to thank all contributors for this new release! In particular,
 Marie Dev Bot,  🙇


### 🍹 Other Improvements

 - [[```ab8fa02a```](https://github.com/marieai/marie-ai/commit/ab8fa02a4be856075c91fcc6c2debbe2e4a26521)] __-__ __release__: working on release (*Marie Dev Bot*)
 - [[```a1a710a0```](https://github.com/marieai/marie-ai/commit/a1a710a03929264eb05e0a7d53a3bda05d3fc9b9)] __-__ __version__: the next version will be 3.0.02 (*Marie Dev Bot*)

<a name=release-note-3-0-03></a>
## Release Note (`3.0.03`)

> Release time: 2023-03-23 15:19:19



🙇 We'd like to thank all contributors for this new release! In particular,
 Marie Dev Bot,  🙇


### 🍹 Other Improvements

 - [[```8bf5135b```](https://github.com/marieai/marie-ai/commit/8bf5135bb76112c01d59bc6e4efee212206904fe)] __-__ __version__: the next version will be 3.0.03 (*Marie Dev Bot*)

<a name=release-note-3-0-04></a>
## Release Note (`3.0.04`)

> Release time: 2023-03-23 21:32:46



🙇 We'd like to thank all contributors for this new release! In particular,
 Marie Dev Bot,  gregbugaj,  Greg,  Scott Christopher Stauffer,  scstauf,  Scott Stauffer,  gbugaj,  🙇


### 🆕 New Features

 - [[```0141e9c3```](https://github.com/marieai/marie-ai/commit/0141e9c3fc1d36e043835d21c2ed3417a2a13876)] __-__ add server to the CLI options (*gregbugaj*)
 - [[```0e979cd8```](https://github.com/marieai/marie-ai/commit/0e979cd837ffd3200ee86a05a45e8f263ec25352)] __-__ add ONNX io bindings (*Greg*)
 - [[```90f9f46f```](https://github.com/marieai/marie-ai/commit/90f9f46fbeff0beb60fe8bc10ead22ce81cec998)] __-__ add bounding box aggregation for horizontal text (*gregbugaj*)
 - [[```96c2e80c```](https://github.com/marieai/marie-ai/commit/96c2e80c3d8363149a0301ecbf2cbadc54156480)] __-__ add support for storing files in s3 storage (*Greg*)
 - [[```c00c411f```](https://github.com/marieai/marie-ai/commit/c00c411f4d4531c4213a17cbac2d72819853e9fe)] __-__ add ssim (*gregbugaj*)
 - [[```01b43776```](https://github.com/marieai/marie-ai/commit/01b4377663528a7be26f36cfd0ce0d755e7bab13)] __-__ add support for messaging and cloud storage (*Greg*)
 - [[```f0625e9d```](https://github.com/marieai/marie-ai/commit/f0625e9df2e36edccbc503478a2d640c4b13a116)] __-__ initial Toast notification service implementation (*gregbugaj*)
 - [[```8481b8f9```](https://github.com/marieai/marie-ai/commit/8481b8f9aa9667b357de58bdcd6a9ab34e4ce9d4)] __-__ merge issue#50 WHitelist iP (*Greg*)
 - [[```c1e59ed8```](https://github.com/marieai/marie-ai/commit/c1e59ed8939d04bec8aff483bb7ee82fe09b2759)] __-__ update to latest TROCR models (*gregbugaj*)
 - [[```38b9404c```](https://github.com/marieai/marie-ai/commit/38b9404c693c3f7105257abdafef7b0d14018726)] __-__ add support for context aware cropping for regions (*gregbugaj*)
 - [[```428e14fe```](https://github.com/marieai/marie-ai/commit/428e14fe400a0c55897725ba488fcbe80dc0b38a)] __-__ initial commit for crop_to_content feature (*gregbugaj*)
 - [[```fa687936```](https://github.com/marieai/marie-ai/commit/fa6879363772886781581f42182a2d17a80f0ad4)] __-__ initial messaging implemenation (*gregbugaj*)
 - [[```4bf151cc```](https://github.com/marieai/marie-ai/commit/4bf151ccbea9239ed686ba3baf2e3a2b5d543d77)] __-__ add executor metadata to results fix issue with device type (*gregbugaj*)
 - [[```a3c70dea```](https://github.com/marieai/marie-ai/commit/a3c70dea0367ca68107e14b04b887946c4f46367)] __-__ add gpu count determination (*gregbugaj*)
 - [[```6ff400bc```](https://github.com/marieai/marie-ai/commit/6ff400bcb3df7bee2cb45b04b32897e6ecbc1b88)] __-__ building container (*Greg*)
 - [[```bc44842c```](https://github.com/marieai/marie-ai/commit/bc44842c882a9f7116f65fcffb7f2c11d86ff076)] __-__ add text extration executors (*gregbugaj*)
 - [[```22ff0761```](https://github.com/marieai/marie-ai/commit/22ff076100c35e4af1e8c3c9d84f2c4d5f40cdde)] __-__ add capability to save tags from the Document (*gregbugaj*)
 - [[```324e0542```](https://github.com/marieai/marie-ai/commit/324e0542919048c35c26de3520e14ccc2ac5155b)] __-__ add support for storing blobs from Document model (*Greg*)
 - [[```b8a42ed3```](https://github.com/marieai/marie-ai/commit/b8a42ed3125408de9655e78e74e8a8c9ecdca0e3)] __-__ Overlay executor impl (*gregbugaj*)
 - [[```9827a0cb```](https://github.com/marieai/marie-ai/commit/9827a0cba140353341a57e408e8cb8df8df749fc)] __-__ added ocrengine and refactor executors (*gregbugaj*)
 - [[```5d368d3e```](https://github.com/marieai/marie-ai/commit/5d368d3e100952f969e256d44aeabf646f02d2aa)] __-__ add support for safely encoding numpy objects fixes issue #45 (*gregbugaj*)
 - [[```092ea79e```](https://github.com/marieai/marie-ai/commit/092ea79ec6a88d5e6cbcff0af8234bde9081c33b)] __-__ add logo (*gregbugaj*)
 - [[```90d4b3f3```](https://github.com/marieai/marie-ai/commit/90d4b3f3791050bff269f6dcb28cd6ff7d102564)] __-__ adding service HTTP REST interface (*gregbugaj*)
 - [[```24e925ca```](https://github.com/marieai/marie-ai/commit/24e925ca59ecf04aa51f5728aab23249834d8e38)] __-__ implementing routes (*gregbugaj*)
 - [[```09f9a215```](https://github.com/marieai/marie-ai/commit/09f9a215df53a746b74d73c5cc77487270883e06)] __-__ wip on server (*Greg*)
 - [[```1ea69a11```](https://github.com/marieai/marie-ai/commit/1ea69a11c4fd3d0d86f5e64f4bbb59c8d6a46a8e)] __-__ __discovery__: consul discovery service,fixed issue with disovery options not visible  resolves #43 (*gregbugaj*)
 - [[```d4f5dd1e```](https://github.com/marieai/marie-ai/commit/d4f5dd1eccd1a370f1f35ac93b859f0c565e241e)] __-__ __discovery__: consul discovery service, resolves #43 (*gregbugaj*)
 - [[```f464cf00```](https://github.com/marieai/marie-ai/commit/f464cf00c24013f07573830ec7c8010381c7ca57)] __-__ work on service disovery (*gregbugaj*)
 - [[```e5fb64dc```](https://github.com/marieai/marie-ai/commit/e5fb64dc31afbe46e7d3c17661c841a46104b462)] __-__ initial work on gateway service discovery #issue-43 (*gbugaj*)
 - [[```7c90a5da```](https://github.com/marieai/marie-ai/commit/7c90a5dae4f470ad7bb48f547dc316160ad8a233)] __-__ __jina__: merging jina (*gbugaj*)
 - [[```8b1440c8```](https://github.com/marieai/marie-ai/commit/8b1440c8d7d10c2200cb7afd81bfee546850e209)] __-__ __formatting__: add code formatting via pre-commit, black, isort, flake8 (*gbugaj*)
 - [[```dda0b4fe```](https://github.com/marieai/marie-ai/commit/dda0b4fe40cf4aa51b76619af7f3745b81154ca1)] __-__ __ci__: Initial release scripts (*gbugaj*)
 - [[```04df7d8b```](https://github.com/marieai/marie-ai/commit/04df7d8bd7c7d92587d2a26e94164226e1b63c12)] __-__ __jina__: Merge Jina changes (*gbugaj*)

### 🐞 Bug fixes

 - [[```1628fadd```](https://github.com/marieai/marie-ai/commit/1628fadd90183019d69a077bcf94e20f27bf8db6)] __-__ issues with the DIT model still using old params (*gregbugaj*)
 - [[```8a6f20c9```](https://github.com/marieai/marie-ai/commit/8a6f20c967154bf57052ae2601b6a25c48d40781)] __-__ added missing await (*gregbugaj*)
 - [[```2e28b21a```](https://github.com/marieai/marie-ai/commit/2e28b21abdf6a981c5e544d35289eb661ba97c1b)] __-__ fix issue with types in python 3.8 (*gregbugaj*)
 - [[```ecc3eb95```](https://github.com/marieai/marie-ai/commit/ecc3eb95a0cbb1c01c6d184894b32ff2b0d771b4)] __-__ fix issue with the &#34;partially initialized module most likely due to circular import&#34; (*Greg*)
 - [[```3602ea9c```](https://github.com/marieai/marie-ai/commit/3602ea9c368ef390691e2e69beef43c9b9dec7ec)] __-__ return status code 503 when gateway is called during dry_run but not yet fully intitialized this is helpfull when registering gateway with service discovery (*gregbugaj*)
 - [[```35d3ef01```](https://github.com/marieai/marie-ai/commit/35d3ef0157802717ced6fdd4defd683547268d12)] __-__ disvoery is done after gateway is initialized (*gregbugaj*)
 - [[```0d9f25e0```](https://github.com/marieai/marie-ai/commit/0d9f25e0819fcda4bcc4d06cdfe4b7dfe0b3ba8b)] __-__ fixed type declarations changed logging levels (*gregbugaj*)
 - [[```71e1eb44```](https://github.com/marieai/marie-ai/commit/71e1eb441a58554d86c2a37cdeb30bc39d113949)] __-__ changed how hash is calculated (*gregbugaj*)
 - [[```1b83e637```](https://github.com/marieai/marie-ai/commit/1b83e637c9e806e78cc967ccf12d6b92981e2216)] __-__ check if frame is on  ndarray type (*gregbugaj*)
 - [[```69f60c37```](https://github.com/marieai/marie-ai/commit/69f60c37a85151d44e5d71b687c8424fe12289d0)] __-__ issue with overlay (*Greg*)
 - [[```01cb5ca0```](https://github.com/marieai/marie-ai/commit/01cb5ca0c4556794fa1b7b4636ed6fb238805900)] __-__ fix issue with 3-channel images not being bursted correctly (*Greg*)
 - [[```2350ae33```](https://github.com/marieai/marie-ai/commit/2350ae33dc18ae2ce44f1a0412f575f3f2e827ac)] __-__ disabled warnings, regen proto (*Greg*)
 - [[```12f99a58```](https://github.com/marieai/marie-ai/commit/12f99a58fc4b4dfca9debfc9543517e47cdd8d86)] __-__ fixes setup script (*Greg*)
 - [[```f2f160fe```](https://github.com/marieai/marie-ai/commit/f2f160febe56b03132c1877a0061d1d00b17bf2e)] __-__ convertsion between frames and documents add stick box checking (*gregbugaj*)
 - [[```e3512dc5```](https://github.com/marieai/marie-ai/commit/e3512dc54058102265ee0a477d43de1965aada0d)] __-__ advertising 0.0.0.0 in service discovery (*gregbugaj*)

### 📗 Documentation

 - [[```51be80d1```](https://github.com/marieai/marie-ai/commit/51be80d10cc8efb64bcba7c9c01efaa421650aa8)] __-__ add example executor (*gbugaj*)
 - [[```6df046af```](https://github.com/marieai/marie-ai/commit/6df046af7fc6395136fa2cb31510d84c61902905)] __-__ add contribution docs (*Greg*)

### 🍹 Other Improvements

 - [[```06285bb3```](https://github.com/marieai/marie-ai/commit/06285bb34584e34b61b8ad44fcd12d21ccf074dc)] __-__ __release__: working on release (*Marie Dev Bot*)
 - [[```10cd3f5a```](https://github.com/marieai/marie-ai/commit/10cd3f5a699fcf76ee1477935cdb9e4e4f8b822b)] __-__ __version__: the next version will be 3.0.04 (*Marie Dev Bot*)
 - [[```8bf5135b```](https://github.com/marieai/marie-ai/commit/8bf5135bb76112c01d59bc6e4efee212206904fe)] __-__ __version__: the next version will be 3.0.03 (*Marie Dev Bot*)
 - [[```a1a710a0```](https://github.com/marieai/marie-ai/commit/a1a710a03929264eb05e0a7d53a3bda05d3fc9b9)] __-__ __version__: the next version will be 3.0.02 (*Marie Dev Bot*)
 - [[```18a4e055```](https://github.com/marieai/marie-ai/commit/18a4e05506bc0ffd9f95833a2bb68590b809f839)] __-__ versioning (*gregbugaj*)
 - [[```9b9ede56```](https://github.com/marieai/marie-ai/commit/9b9ede56ba96f47d80b009f8efcdb25e513548d9)] __-__ integrating marie_server (*gregbugaj*)
 - [[```6a1df1cf```](https://github.com/marieai/marie-ai/commit/6a1df1cf79df3ba22a1f00fdb4971a31932dcf27)] __-__ WIP (*Greg*)
 - [[```68163a77```](https://github.com/marieai/marie-ai/commit/68163a7725dda7184d4dfbf7ea58954fd8f8c1c9)] __-__ optimization (*Greg*)
 - [[```cee48442```](https://github.com/marieai/marie-ai/commit/cee484425ea45bb8b1cdd033273d2f39b43c1e8e)] __-__ onnx integration for overlay (*gregbugaj*)
 - [[```cdbc6515```](https://github.com/marieai/marie-ai/commit/cdbc651535d819ff6c8e7c14ce6759974ab95aea)] __-__ performance improvments (*gregbugaj*)
 - [[```05be3b19```](https://github.com/marieai/marie-ai/commit/05be3b19523f0683822b6ac3a022a4cbfaf81582)] __-__ onnx conversion (*Greg*)
 - [[```5f27d642```](https://github.com/marieai/marie-ai/commit/5f27d642dcdf6d7b730dd0a32f6b007e0ec52a1d)] __-__ convertion to onnx runtime (*gregbugaj*)
 - [[```bae8c657```](https://github.com/marieai/marie-ai/commit/bae8c657511fed9a0a05f9e1a24d51c7afb029be)] __-__ notes (*Greg*)
 - [[```6670e261```](https://github.com/marieai/marie-ai/commit/6670e261b0923c9ad650fb6085e83c7a76c73b3d)] __-__ pix2pix model conversion and inference (*Greg*)
 - [[```422c9c84```](https://github.com/marieai/marie-ai/commit/422c9c84567b6bfdcdde69fcebebade25da5c582)] __-__ optimizations (*gregbugaj*)
 - [[```9de4c9b7```](https://github.com/marieai/marie-ai/commit/9de4c9b76e1643f3e0036b44317b172194547bc6)] __-__ overlay cleanup (*gregbugaj*)
 - [[```80a71bcb```](https://github.com/marieai/marie-ai/commit/80a71bcb14683b50d096ef1d698948e025f8d745)] __-__ bouding box aggregation (*gregbugaj*)
 - [[```cb639ede```](https://github.com/marieai/marie-ai/commit/cb639ede2e98ce096751da8cf9542db1e9326c35)] __-__ pix2pix (*Greg*)
 - [[```f09bf869```](https://github.com/marieai/marie-ai/commit/f09bf869fb538aaedbf459c0ac92cfed885862e7)] __-__ cleanup and setup of s3 storage (*gregbugaj*)
 - [[```bd09b4b8```](https://github.com/marieai/marie-ai/commit/bd09b4b8130186a149605ccaf1425a876c130fe0)] __-__ Migration to Pytorch2 and Python 3.11 (*gregbugaj*)
 - [[```c34cd377```](https://github.com/marieai/marie-ai/commit/c34cd377ce1ffe67a5e405e3e3e93c6f74ee121f)] __-__ cleanup of service discovery (*Greg*)
 - [[```f190762f```](https://github.com/marieai/marie-ai/commit/f190762ff569eebdf9c36c0531f4ac9b20c21911)] __-__ work on implementing storage in overlay processor (*gregbugaj*)
 - [[```a150c94d```](https://github.com/marieai/marie-ai/commit/a150c94dd24dbf53b53ad450564656a5f205de8f)] __-__ changed logging level (*gregbugaj*)
 - [[```1b6dd9e2```](https://github.com/marieai/marie-ai/commit/1b6dd9e2053cc290cead0b52995b132870c8c87c)] __-__ discovery service refactor (*gregbugaj*)
 - [[```92d76166```](https://github.com/marieai/marie-ai/commit/92d761665e4f684f3fa49b4f2dbf7c67e5dc8d9a)] __-__ serialization (*gregbugaj*)
 - [[```c6a6c2c6```](https://github.com/marieai/marie-ai/commit/c6a6c2c669dad25e8a50072a303d2a16d3f78b85)] __-__ missing requirements (*gregbugaj*)
 - [[```70aabe04```](https://github.com/marieai/marie-ai/commit/70aabe04510ce390839368c13f3a7382f22a79e7)] __-__ refactoring (*gregbugaj*)
 - [[```65dd7b74```](https://github.com/marieai/marie-ai/commit/65dd7b74ff8e58eff63615e6ea1d885fd4039a78)] __-__ implementing storage (*gregbugaj*)
 - [[```7cfbf7dd```](https://github.com/marieai/marie-ai/commit/7cfbf7dd4e3bf98a9eb26e379ed508e415706cb6)] __-__ add utilities (*gregbugaj*)
 - [[```1e561096```](https://github.com/marieai/marie-ai/commit/1e56109618a7b8b87ee0217f46f1f3efaf4f7699)] __-__ cleanup (*gregbugaj*)
 - [[```cf7ccde5```](https://github.com/marieai/marie-ai/commit/cf7ccde54fa3268b1212fadb6bdba224828e556d)] __-__ WIP, notes (*gregbugaj*)
 - [[```e8c3bc41```](https://github.com/marieai/marie-ai/commit/e8c3bc41edfab4c8a7fcf33902298b774e3038fe)] __-__ container build (*gregbugaj*)
 - [[```e7cfb25e```](https://github.com/marieai/marie-ai/commit/e7cfb25ea91bb3d333353ad46b63e1c0b113f84c)] __-__ cicd (*gregbugaj*)
 - [[```4d397c89```](https://github.com/marieai/marie-ai/commit/4d397c89609cf8f269509ed9f64a9adc5fad70b7)] __-__ build system (*gregbugaj*)
 - [[```5dc271da```](https://github.com/marieai/marie-ai/commit/5dc271dae8366c32f64e69ff0f7299808601534b)] __-__ cleanup overlay executors (*gregbugaj*)
 - [[```7986199e```](https://github.com/marieai/marie-ai/commit/7986199e634d706b6b219ae56192ad84d25d8612)] __-__ refactoroing overlay processor (*gregbugaj*)
 - [[```962e294d```](https://github.com/marieai/marie-ai/commit/962e294d41e90fe39ebd7c2b5d80df5302053647)] __-__ port Named Enity Recogition executor to use Executors (*gregbugaj*)
 - [[```5b0fc6b5```](https://github.com/marieai/marie-ai/commit/5b0fc6b55928cf1bea91419d2ad250ba9248ab75)] __-__ converting executors (*gregbugaj*)
 - [[```e33abc4e```](https://github.com/marieai/marie-ai/commit/e33abc4ee040baaf6e2d231c9d5dd709b2e2de5e)] __-__ work on HTTP gateway (*Greg*)
 - [[```92feca11```](https://github.com/marieai/marie-ai/commit/92feca11f5781effdc8e0195099b89fdc9207a4b)] __-__ work on docker build (*gregbugaj*)
 - [[```644e9166```](https://github.com/marieai/marie-ai/commit/644e9166063988c5981513335d69e2d11fdae6e1)] __-__ wip on container (*gregbugaj*)
 - [[```33aadb4b```](https://github.com/marieai/marie-ai/commit/33aadb4bed5693463195e7896d8064dc18f0dab6)] __-__ __jina__: merged some tests (*gbugaj*)
 - [[```68d28077```](https://github.com/marieai/marie-ai/commit/68d2807759074d8a6bffb2e74fead182fadeecfd)] __-__ __jina__: merge latest changes and add integration tests (*gbugaj*)
 - [[```00697ede```](https://github.com/marieai/marie-ai/commit/00697edefdffa4b36b9315d42dead3dfd3ac0291)] __-__ __jina__: merge latest changes and add itnergration tests (*gbugaj*)
 - [[```c6d4a806```](https://github.com/marieai/marie-ai/commit/c6d4a8066776311996d8ee3ebc5c1873ddcd4ee7)] __-__ __jina__: merge hubble (*gbugaj*)
 - [[```a14ca598```](https://github.com/marieai/marie-ai/commit/a14ca598c2b5943f16a367349a950329984f53bb)] __-__ __jina__: merge (*gbugaj*)
 - [[```36050d27```](https://github.com/marieai/marie-ai/commit/36050d27eaed3d5d8ee3634ab815a2b36e9e0d86)] __-__ __jina__: merge latest changes (*Greg*)
 - [[```9ae2f760```](https://github.com/marieai/marie-ai/commit/9ae2f7603153814f92dea3735d2aaf96f5e68a0f)] __-__ __jina__: merge jina changes (*gbugaj*)

<a name=release-note-3-0-21></a>
## Release Note (`3.0.21`)

> Release time: 2023-11-01 15:00:03



🙇 We'd like to thank all contributors for this new release! In particular,
 Marie Dev Bot,  Greg,  🙇


### 🐞 Bug fixes

 - [[```db35b812```](https://github.com/marieai/marie-ai/commit/db35b8125e9c1b65e7af776ec4b28b51e56d2ad3)] __-__ add pipeline device (cpu,gpu) handling (*Marie Dev Bot*)

### 🍹 Other Improvements

 - [[```39e294a2```](https://github.com/marieai/marie-ai/commit/39e294a2a9d836f65054f3f89333276ba17be980)] __-__ classification (*Greg*)
 - [[```801b5cd6```](https://github.com/marieai/marie-ai/commit/801b5cd6b6bde6eb0422ea6de0e4eb9c2e3df453)] __-__ classfication (*Greg*)
 - [[```7ad42803```](https://github.com/marieai/marie-ai/commit/7ad42803b006add6ced0b425a6ba23c15066f560)] __-__ wip (*Greg*)
 - [[```5660085e```](https://github.com/marieai/marie-ai/commit/5660085e5c3b31a29f8771e24f8a19fd02be7b14)] __-__ document classification (*Marie Dev Bot*)
 - [[```cc2848fc```](https://github.com/marieai/marie-ai/commit/cc2848fcf957d253302f175a2552e9e76d6de8c9)] __-__ integrating document classfication (*Marie Dev Bot*)
 - [[```cba3a380```](https://github.com/marieai/marie-ai/commit/cba3a3800a9771a7b7843c76e5d0f8d0dce36d00)] __-__ cleanup (*Marie Dev Bot*)
 - [[```7f4a128c```](https://github.com/marieai/marie-ai/commit/7f4a128c72de3894406a669b8a82315581fcfa97)] __-__ __release__: fixed release version 3.0.21 to match container versions (*Marie Dev Bot*)

<a name=release-note-3-0-21></a>
## Release Note (`3.0.21`)

> Release time: 2023-11-01 15:03:52



🙇 We'd like to thank all contributors for this new release! In particular,
 Marie Dev Bot,  🙇


### 🍹 Other Improvements

 - [[```368fbfb4```](https://github.com/marieai/marie-ai/commit/368fbfb46912418d88df33a1ec7039b864cec391)] __-__ __release__: release v3.0.21 (*Marie Dev Bot*)

