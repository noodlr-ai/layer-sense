import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import { buildTrainingData, encodeSequences, padSequences, vocab } from './domain';
import { trainingData } from './trainingData';

function createModel(vocabSize: number, embeddingDim: number, lstmUnits: number) {
    const model = tf.sequential();
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: embeddingDim,
        maskZero: true // Ignore padding values which are set to 0
    }));
    model.add(tf.layers.lstm({
        units: lstmUnits,
        returnSequences: false
    }));
    model.add(tf.layers.dense({
        units: vocabSize, // Predict only valid layers, excluding PAD
        activation: 'softmax' // Output probabilities
    }));
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return model;
}

// prepareTrainingData prepares the training data for the model, encoding  the sequences into indices, adding paddings, and one-hot encoding the targets
function prepareTrainingData(): { maxLen: number, tfXTrain: tf.Tensor2D, tfYTrain: tf.Tensor2D } {
    // Encode sequences into indices
    const encodedSequences = encodeSequences(trainingData);

    // Build training data and get max sequence length for padding
    const { xTrain, yTrain } = buildTrainingData(encodedSequences);

    // Note: maxLen is required during training to ensure the tensors are the same size.The sequence input needs to be at least 2, even if the second element is just PAD !important!
    const maxLen = Math.max(...xTrain.map(x => x.length), 2);
    // Pad sequences to the same length
    const xTrainPadded = padSequences(xTrain, maxLen);

    // One-hot encode targets
    const yTrainOHE: number[][] = yTrain.map<number[]>(idx => {
        const oneHot: number[] = new Array(vocab.length).fill(0);
        // const oneHot: number[] = new Array(vocab.length - 1).fill(0);
        oneHot[idx] = 1;
        // oneHot[idx - 1] = 1;
        return oneHot;
    });

    // Convert to tensors and return
    return {
        maxLen,
        tfXTrain: tf.tensor2d(xTrainPadded),
        tfYTrain: tf.tensor2d(yTrainOHE)
    }
}

// Prepare the training data
const { maxLen, tfXTrain, tfYTrain } = prepareTrainingData();

// Create and train the model
const model = createModel(vocab.length, 16, 32);
await model.fit(tfXTrain, tfYTrain, {
    epochs: 50,
    batchSize: 4,
    validationSplit: 0.2,
    // verbose: 0, // Suppress training output
});
console.log('Model training complete; saving model to disk');
await model.save('file://./build/model');

// Save our vocabulary and maxLen for later use
const metadata = { vocab, maxLen };
fs.writeFileSync('build/model/model-metadata.json', JSON.stringify(metadata));