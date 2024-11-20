import * as tf from '@tensorflow/tfjs';

// const layerTypes = ["Conv2D", "MaxPooling", "Dense", "Dropout", "Flatten", "BatchNormalization"] as const;
const layerTypes = ["Dense", "Dropout", "BatchNormalization", "Training"] as const;
const activationTypes = ["linear", "relu", "sigmoid", "softmax", "tanh"] as const;
const lossFunctions = ["mse", "mae", "categoricalCrossentropy", "binaryCrossentropy"] as const;
type LayerType = typeof layerTypes[number];

type LayerBase = {
    type: LayerType;
}

type DenseLayer = LayerBase & {
    type: "Dense";
    units: number;
    activation: typeof activationTypes[number];
}

type DropoutLayer = LayerBase & {
    type: "Dropout";
    rate: number;
}

type BatchNormalizationLayer = LayerBase & {
    type: "BatchNormalization";
}

type TrainingLayer = LayerBase & {
    type: "Training";
    lossFunction: typeof lossFunctions[number];
}

type Layer = DenseLayer | DropoutLayer | BatchNormalizationLayer | TrainingLayer;

const createDenseLayer = (units: number, activation: typeof activationTypes[number]): DenseLayer => ({ type: 'Dense', activation, units, });
const createDropoutLayer = (rate: number): DropoutLayer => ({ type: 'Dropout', rate });
const createBatchNormalizationLayer = (): BatchNormalizationLayer => ({ type: 'BatchNormalization' });
const createTrainingLayer = (lossFunction: typeof lossFunctions[number]): TrainingLayer => ({ type: 'Training', lossFunction });

const featuresMap = {
    layerType: layerTypes.length,
    activationType: activationTypes.length,
    units: 1,
    rate: 1,
    lossFunction: lossFunctions.length
};

const featuresMapLength = Object.entries(featuresMap).map(([key, value]) => value).reduce((a, b) => a + b);

function indexOfFeature(feature: keyof typeof featuresMap) {
    switch (feature) {
        case 'layerType': {
            return 0;
        }
        case 'activationType': {
            return featuresMap.layerType;
        }
        case 'units': {
            return featuresMap.layerType + featuresMap.activationType;
        }
        case 'rate': {
            return featuresMap.layerType + featuresMap.activationType + featuresMap.units;
        }
        case 'lossFunction': {
            return featuresMap.layerType + featuresMap.activationType + featuresMap.units + featuresMap.rate;
        }
    }
}

function createFeatureFromLayer(layer: Layer) {
    let feature = Array(featuresMapLength).fill(0) as number[];
    feature[layerTypes.indexOf(layer.type)] = 1;
    switch (layer.type) {
        case 'Dense': {
            feature[activationTypes.indexOf(layer.activation) + indexOfFeature('activationType')] = 1;
            feature[indexOfFeature('units')] = layer.units;
            return feature;
        }
        case 'Dropout': {
            feature[indexOfFeature('rate')] = layer.rate;
            return feature
        }
        case 'BatchNormalization': {
            return feature;
        }
        case 'Training': {
            feature[lossFunctions.indexOf(layer.lossFunction) + indexOfFeature('lossFunction')] = 1;
            return feature;
        }
    }
}

function convertFeaturesToLayer(feature: number[]): Layer {
    const layerType = layerTypes[feature.indexOf(1)]; // find the first "1" in the feature array, which is the layer type
    switch (layerType) {
        case 'Dense': {
            const idx = indexOfFeature('activationType');
            const activation = activationTypes[feature.indexOf(1, idx) - idx]; // find the first "1" after the activationType index
            const units = feature[indexOfFeature('units')];
            return createDenseLayer(units, activation);
        }
        case 'Dropout': {
            const rate = feature[indexOfFeature('rate')];
            return createDropoutLayer(rate);
        }
        case 'BatchNormalization': {
            return createBatchNormalizationLayer();
        }
        case 'Training': {
            const idx = indexOfFeature('lossFunction');
            const lossFunction = lossFunctions[feature.indexOf(1, idx) - idx]; // find the first "1" after the lossFunction index
            return createTrainingLayer(lossFunction);
        }
    }
}

function encodeLayerSequences(layerSequences: Layer[][]): number[][][] {
    return layerSequences.map(seq => seq.map(layer => createFeatureFromLayer(layer)));
}

const layerSequences: Layer[][] = [
    [createDenseLayer(64, 'relu'), createDropoutLayer(0.5), createBatchNormalizationLayer(), createDenseLayer(1, 'linear'), createTrainingLayer('mse')],
]

// Note: this can be a reduce() function, but it looks terrible
// const encodedSequences: number[][][] = []
// layerSequences.forEach(seq => {
//     encodedSequences.push(seq.map(layer => createFeatureFromLayer(layer)));
// });

// console.log(encodedSequences);
const encodedSequences = encodeLayerSequences(layerSequences);


// This function splits a sequence into multiple input-output pairs for testing based
// Each sequence will result it in multiple input-output pairs
function buildTrainingData(encodedSequences: number[][][]) {
    // Prepare input-output pairs
    let xTrain: number[][][] = [];
    let yTrain: number[][] = [];
    encodedSequences.forEach(seq => {
        for (let t = 0; t < seq.length - 1; t++) {
            xTrain.push(seq.slice(0, t + 1)); // Input sequence
            yTrain.push(seq[t + 1]);         // Target layer
        }
    });
    return { xTrain, yTrain };
}

const { xTrain, yTrain } = buildTrainingData(encodedSequences);
console.log(xTrain, yTrain);

// LEFT-OFF: I am not sure if I need the same number of sequences?
// LEFT-OFF: I am not sure how training will work with this data.
// LEFT-OFF: Ask ChatGPT with dummy data to start

// padSequences ensures all of the x-values are the same length
function padSequences(sequences: number[][]) {
    const maxLen = Math.max(...xTrain.map(x => x.length), 2); // Note: this is very important, the sequence input needs to be at least 2, even if the second element is PAD !important!
    const paddedSequences = sequences.map(seq => {
        const pad = new Array(maxLen - seq.length).fill(layerToIdx["PAD"]);
        return [...seq, ...pad];
    });
    return { maxLen, paddedSequences };
}

// const { xTrain, yTrain } = buildTrainingData(encodedSequences);

// Pad sequences to the same length
// const { maxLen, paddedSequences: xTrainPadded } = padSequences(xTrain);


const tfXTrain = tf.tensor3d(xTrain);
const tfYTrain = tf.tensor2d(yTrain);


// function createModel(inputDim: number, lstmUnits: number) {
//     const model = tf.sequential();
//     model.add(tf.layers.lstm({
//         inputShape: [null, inputDim],
//         units: lstmUnits,
//         returnSequences: false,
//     }));
//     model.add(tf.layers.dense({
//         units: inputDim, // Predict the next layer's features
//         activation: 'linear'
//     }));
//     model.compile({
//         optimizer: 'adam',
//         loss: 'meanSquaredError', // Mean squared error for feature prediction
//         metrics: ['mae'] // Mean absolute error
//     });
//     return model;
// }

// function createModel(inputDim: number, lstmUnits: number) {
//     const model = tf.sequential();
//     model.add(tf.layers.lstm({
//         inputShape: [null, inputDim],
//         units: lstmUnits,
//         returnSequences: false,
//     }));
//     model.add(tf.layers.dense({
//         units: inputDim, // Predict the next layer's features
//         activation: 'softmax'
//     }));
//     model.compile({
//         optimizer: 'adam',
//         loss: 'categoricalCrossentropy', // Mean squared error for feature prediction
//         metrics: ['accuracy'] // Mean absolute error
//     });
//     return model;
// }

function createModel(inputDim: number, lstmUnits: number, numLayerTypes: number, numActivations: number, maxUnits: number, maxRate: number, numLossFunctions: number) {
    const input = tf.input({ shape: [null, inputDim] });

    // LSTM layer
    const lstm = tf.layers.lstm({
        units: lstmUnits,
        returnSequences: false
    }).apply(input);

    // Layer type prediction (softmax for categorical prediction)
    const layerType = tf.layers.dense({
        units: numLayerTypes,
        activation: 'softmax',
        name: 'layer_type'
    }).apply(lstm);

    // Activation function prediction (softmax for categorical prediction)
    const activationFunction = tf.layers.dense({
        units: numActivations,
        activation: 'softmax',
        name: 'activation_function'
    }).apply(lstm);

    // Units prediction (sigmoid for range constraint, scaled by maxUnits)
    const units = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        name: 'units'
    }).apply(lstm);

    // Rate prediction (sigmoid for range constraint, scaled by maxRate)
    const rate = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        name: 'rate'
    }).apply(lstm);

    // Loss function prediction (softmax for categorical prediction)
    const lossFunction = tf.layers.dense({
        units: numLossFunctions,
        activation: 'softmax',
        name: 'loss_function'
    }).apply(lstm);

    // Create the model with multiple outputs
    // Create the model with multiple outputs
    const model = tf.model({
        inputs: input,
        outputs: {
            layer_type: layerType,
            activation_function: activationFunction,
            units: units,
            rate: rate,
            loss_function: lossFunction
        }
    });

    // Compile the model with a loss for each output
    model.compile({
        optimizer: 'adam',
        loss: {
            layer_type: 'categoricalCrossentropy',
            activation_function: 'categoricalCrossentropy',
            units: 'meanSquaredError',
            rate: 'meanSquaredError',
            loss_function: 'categoricalCrossentropy',
        },
        metrics: ['accuracy']
    });

    return model;
}

// LEFT-OFF: it would seem that the concept is correct, that I will want multiple outputs
// LEFT-OFF: figure out if I do need to have multiple outputs and how to model them properly.
// LEFT-OFF: if I need multiple outputs then it may create problems with outputs being inconsistent for the layer type
// LEFT-OFF: maybe I will need multiple models for each output type???


// Create and train the model
// const model = createModel(featuresMapLength, 32);
const model = createModel(featuresMapLength, 32, featuresMap.layerType, featuresMap.activationType, 128, 0.9, featuresMap.lossFunction);
// console.log(model);

await model.fit(tfXTrain, tfYTrain, {
    epochs: 50,
    batchSize: 2,
    validationSplit: 0.2
});
console.log('Model has been trained');


// const testSeq = [createDenseLayer(64, 'relu'), createDropoutLayer(0.5), createBatchNormalizationLayer()];
const testSeq = [createDenseLayer(64, 'relu'), createDropoutLayer(0.5)];
const encodedTestSeq = encodeLayerSequences([testSeq, testSeq, testSeq]);
const tfTestSeq = tf.tensor3d(encodedTestSeq);

const probs = model.predict(tfTestSeq);

function displaySequence(probs: tf.Tensor | tf.Tensor[]) {
    console.log('Predicted Probabilities');
    if (Array.isArray(probs)) {
        // const p = probs[0].squeeze().arraySync();
        // console.log(p);

        console.log(probs.map(p => p.arraySync()));

    } else {
        console.log('single array');
        console.log(probs.arraySync());
        // const p = probs.squeeze().arraySync() as number[];
        // console.log(p);
        // // We add 1 to skip the padding index
        // console.log(p.map((prob, idx) => `${idxToLayer[idx]}: ${prob.toFixed(3)}`).join('\n'));
        // console.log(p);
    }
}

displaySequence(probs);

// LEFT-OFF: this runs
// LEFT-OFF: the results are the same across the three sequences
// LEFT-OFF: the results vary wildly between runs
// LEFT-OFF: really not sure what the output even means. They aren't probabilities.