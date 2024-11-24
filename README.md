# Noodlr Layer Sense

Layer sense predicts the next layer in a sequence when developing a workflow in Noodlr.

# Development

`npm run build`
To build the .ts files into .js

`npm run test-layer-sequence`
To run tests for layer sequencing

`node build/trainModel.js`
To train the model and output the model to 'build/model'

# TFJS-Node Error

There is an issue with running the package on Windows 11 that requires the following workaround:

https://github.com/tensorflow/tfjs/issues/8176

`cp node_modules/\@tensorflow/tfjs-node/deps/lib/tensorflow.dll node_modules/\@tensorflow/tfjs-node/lib/napi-v8/`

Alternatively, you can run `node fix-tensorflow`, which will run the utility file found at `utils/fixTensorflowJSNode.js`
