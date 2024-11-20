import * as tf from "@tensorflow/tfjs";
const layerTypes = ["Dense", "Dropout", "BatchNormalization", "Training"];
const activationTypes = ["linear", "relu", "sigmoid", "softmax", "tanh"];
const lossFunctions = ["mse", "mae", "categoricalCrossentropy", "binaryCrossentropy"];
const createDenseLayer = (units, activation) => ({ type: "Dense", activation, units });
const createDropoutLayer = (rate) => ({ type: "Dropout", rate });
const createBatchNormalizationLayer = () => ({ type: "BatchNormalization" });
const createTrainingLayer = (lossFunction) => ({ type: "Training", lossFunction });
const featuresMap = {
  layerType: layerTypes.length,
  activationType: activationTypes.length,
  units: 1,
  rate: 1,
  lossFunction: lossFunctions.length
};
const featuresMapLength = Object.entries(featuresMap).map(([key, value]) => value).reduce((a, b) => a + b);
function indexOfFeature(feature) {
  switch (feature) {
    case "layerType": {
      return 0;
    }
    case "activationType": {
      return featuresMap.layerType;
    }
    case "units": {
      return featuresMap.layerType + featuresMap.activationType;
    }
    case "rate": {
      return featuresMap.layerType + featuresMap.activationType + featuresMap.units;
    }
    case "lossFunction": {
      return featuresMap.layerType + featuresMap.activationType + featuresMap.units + featuresMap.rate;
    }
  }
}
function createFeatureFromLayer(layer) {
  let feature = Array(featuresMapLength).fill(0);
  feature[layerTypes.indexOf(layer.type)] = 1;
  switch (layer.type) {
    case "Dense": {
      feature[activationTypes.indexOf(layer.activation) + indexOfFeature("activationType")] = 1;
      feature[indexOfFeature("units")] = layer.units;
      return feature;
    }
    case "Dropout": {
      feature[indexOfFeature("rate")] = layer.rate;
      return feature;
    }
    case "BatchNormalization": {
      return feature;
    }
    case "Training": {
      feature[lossFunctions.indexOf(layer.lossFunction) + indexOfFeature("lossFunction")] = 1;
      return feature;
    }
  }
}
function convertFeaturesToLayer(feature) {
  const layerType = layerTypes[feature.indexOf(1)];
  switch (layerType) {
    case "Dense": {
      const idx = indexOfFeature("activationType");
      const activation = activationTypes[feature.indexOf(1, idx) - idx];
      const units = feature[indexOfFeature("units")];
      return createDenseLayer(units, activation);
    }
    case "Dropout": {
      const rate = feature[indexOfFeature("rate")];
      return createDropoutLayer(rate);
    }
    case "BatchNormalization": {
      return createBatchNormalizationLayer();
    }
    case "Training": {
      const idx = indexOfFeature("lossFunction");
      const lossFunction = lossFunctions[feature.indexOf(1, idx) - idx];
      return createTrainingLayer(lossFunction);
    }
  }
}
function encodeLayerSequences(layerSequences2) {
  return layerSequences2.map((seq) => seq.map((layer) => createFeatureFromLayer(layer)));
}
const layerSequences = [
  [createDenseLayer(64, "relu"), createDropoutLayer(0.5), createBatchNormalizationLayer(), createDenseLayer(1, "linear"), createTrainingLayer("mse")]
];
const encodedSequences = encodeLayerSequences(layerSequences);
function buildTrainingData(encodedSequences2) {
  let xTrain2 = [];
  let yTrain2 = [];
  encodedSequences2.forEach((seq) => {
    for (let t = 0; t < seq.length - 1; t++) {
      xTrain2.push(seq.slice(0, t + 1));
      yTrain2.push(seq[t + 1]);
    }
  });
  return { xTrain: xTrain2, yTrain: yTrain2 };
}
const { xTrain, yTrain } = buildTrainingData(encodedSequences);
console.log(xTrain, yTrain);
function padSequences(sequences) {
  const maxLen = Math.max(...xTrain.map((x) => x.length), 2);
  const paddedSequences = sequences.map((seq) => {
    const pad = new Array(maxLen - seq.length).fill(layerToIdx["PAD"]);
    return [...seq, ...pad];
  });
  return { maxLen, paddedSequences };
}
const tfXTrain = tf.tensor3d(xTrain);
const tfYTrain = tf.tensor2d(yTrain);
function createModel(inputDim, lstmUnits, numLayerTypes, numActivations, maxUnits, maxRate, numLossFunctions) {
  const input = tf.input({ shape: [null, inputDim] });
  const lstm = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: false
  }).apply(input);
  const layerType = tf.layers.dense({
    units: numLayerTypes,
    activation: "softmax",
    name: "layer_type"
  }).apply(lstm);
  const activationFunction = tf.layers.dense({
    units: numActivations,
    activation: "softmax",
    name: "activation_function"
  }).apply(lstm);
  const units = tf.layers.dense({
    units: 1,
    activation: "sigmoid",
    name: "units"
  }).apply(lstm);
  const rate = tf.layers.dense({
    units: 1,
    activation: "sigmoid",
    name: "rate"
  }).apply(lstm);
  const lossFunction = tf.layers.dense({
    units: numLossFunctions,
    activation: "softmax",
    name: "loss_function"
  }).apply(lstm);
  const model2 = tf.model({
    inputs: input,
    outputs: {
      layer_type: layerType,
      activation_function: activationFunction,
      units,
      rate,
      loss_function: lossFunction
    }
  });
  model2.compile({
    optimizer: "adam",
    loss: {
      layer_type: "categoricalCrossentropy",
      activation_function: "categoricalCrossentropy",
      units: "meanSquaredError",
      rate: "meanSquaredError",
      loss_function: "categoricalCrossentropy"
    },
    metrics: ["accuracy"]
  });
  return model2;
}
const model = createModel(featuresMapLength, 32, featuresMap.layerType, featuresMap.activationType, 128, 0.9, featuresMap.lossFunction);
await model.fit(tfXTrain, tfYTrain, {
  epochs: 50,
  batchSize: 2,
  validationSplit: 0.2
});
console.log("Model has been trained");
const testSeq = [createDenseLayer(64, "relu"), createDropoutLayer(0.5)];
const encodedTestSeq = encodeLayerSequences([testSeq, testSeq, testSeq]);
const tfTestSeq = tf.tensor3d(encodedTestSeq);
const probs = model.predict(tfTestSeq);
function displaySequence(probs2) {
  console.log("Predicted Probabilities");
  if (Array.isArray(probs2)) {
    console.log(probs2.map((p) => p.arraySync()));
  } else {
    console.log("single array");
    console.log(probs2.arraySync());
  }
}
displaySequence(probs);
