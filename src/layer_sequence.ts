import * as tf from '@tensorflow/tfjs';

// Vocabulary of layers
const vocab = ['PAD', 'Dataset', 'RowSplit', 'ColumnSplit', 'ColumnSplice', 'Dense', 'Dropout', 'BatchNormalization', 'Training', 'Launcher', 'ManualDataEntry', 'DataViewer', 'Transformer']
// Example training sequences
const sequences = [
    ['Launcher', 'ManualDataEntry', 'Transformer', 'DataViewer'],
    ['Launcher', 'ManualDataEntry', 'Transformer', 'DataViewer'],
    ['Launcher', 'ManualDataEntry', 'Transformer', 'Transformer', 'DataViewer'],
    ['Dataset', 'RowSplit', 'ColumnSplit', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplit', 'Dense', 'Dense', 'Dense', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplit', 'Dense', 'Dense', 'Dense', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplit', 'Dense', 'Dense', 'Dense', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplit', 'Dense', 'Dense', 'Dense', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplice', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplice', 'Dense', 'Dense', 'Dense', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplice', 'Dense', 'Dense', 'Dense', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplice', 'Dense', 'Dense', 'Dense', 'Training'],
    ['Dataset', 'RowSplit', 'ColumnSplice', 'Dense', 'Dense', 'Dense', 'Training'],
];

// Note: the model input will be the maximum sequences.length - 1

// Turns a sequence of layers into a sequence of indices
function encodeSequences(sequences: string[][], vocab: string[]) {
    const layerToIdx = Object.fromEntries(vocab.map((layer, idx) => [layer, idx]));
    return sequences.map(seq => seq.map(layer => layerToIdx[layer]));
}

// This function splits a sequence into multiple input-output pairs for testing based
// Each sequence will result it in multiple input-output pairs
function buildTrainingData(encodedSequences: number[][]) {
    // Prepare input-output pairs
    let xTrain: number[][] = [];
    let yTrain: number[] = [];
    encodedSequences.forEach(seq => {
        for (let t = 0; t < seq.length - 1; t++) {
            xTrain.push(seq.slice(0, t + 1)); // Input sequence
            yTrain.push(seq[t + 1]);         // Target layer
        }
    });
    return { xTrain, yTrain };
}

function buildIndices(vocab: string[]) {
    const layerToIdx = Object.fromEntries(vocab.map((layer, idx) => [layer, idx]));
    const idxToLayer = Object.fromEntries(vocab.map((layer, idx) => [idx, layer]));
    return { layerToIdx, idxToLayer };
}

function padSequence(sequence: number[], maxLen: number) {
    const pad = new Array(maxLen - sequence.length).fill(layerToIdx['PAD']);
    return [...sequence, ...pad];
}

// padSequences ensures all of the x-values are the same length
function padSequences(sequences: number[][], maxLen: number) {
    return sequences.map(seq => padSequence(seq, maxLen));
}

// Build indices for encoding and decoding
const { layerToIdx, idxToLayer } = buildIndices(vocab);

// Encode sequences into indices
const encodedSequences = encodeSequences(sequences, vocab);

// Build training data and get max sequence length for padding
const { xTrain, yTrain } = buildTrainingData(encodedSequences);
const maxLen = Math.max(...xTrain.map(x => x.length), 2); // Note: this is very important, the sequence input needs to be at least 2, even if the second element is PAD !important!

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

// Convert to tensors
const tfXTrain = tf.tensor2d(xTrainPadded);
const tfYTrain = tf.tensor2d(yTrainOHE);
// console.log(tfXTrain.print());
// console.log(tfYTrain.print());
// console.log(tfXTrain.shape);
// console.log(tfYTrain.shape);

// Build and train the model
function createModel(vocabSize: number, embeddingDim: number, lstmUnits: number) {
    const model = tf.sequential();
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: embeddingDim,
        maskZero: true // Ignore padding values
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
        metrics: ['accuracy']
    });
    return model;
}

// Create and train the model
const model = createModel(vocab.length, 16, 32);
console.log('Training model');
await model.fit(tfXTrain, tfYTrain, {
    epochs: 50,
    batchSize: 2,
    validationSplit: 0.2
});
console.log('Model training complete');

// const testSeq = [layerToIdx['Conv2D'], layerToIdx['MaxPooling'], layerToIdx['Flatten']];
// const testSeq = [layerToIdx['Dense']];
// const testSeq = [layerToIdx['Dropout']];
// const testSeq = [layerToIdx['Conv2D'], layerToIdx['MaxPooling']];
// const testSeq = [layerToIdx['Conv2D']];
// const testSeq = [layerToIdx['Flatten']];
// const testSeq = [layerToIdx['Dense'], layerToIdx['Dropout']];
// const testSeq = [layerToIdx['Dropout'], layerToIdx['Dense']];
// const inputSeq = testSeq.map(layer => layerToIdx[layer]);

const testSeq = [layerToIdx['Launcher'], layerToIdx['ManualDataEntry']];
const paddedSeq = padSequence(testSeq, maxLen);
const tfTestSeq = tf.tensor2d([paddedSeq]);

function displayProbabilities(probs: tf.Tensor | tf.Tensor[]) {
    console.log('Predicted Probabilities');
    if (Array.isArray(probs)) {
        const p = probs[0].squeeze().arraySync();
        console.log(p);

    } else {
        const p = probs.squeeze().arraySync() as number[];
        // We add 1 to skip the padding index
        console.log(p.map((prob, idx) => `${idxToLayer[idx]}: ${prob.toFixed(3)}`).join('\n'));
    }
}

const probs = model.predict(tfTestSeq);
displayProbabilities(probs);

// Implement Tree Search for Sequence Prediction
async function treeSearch(model: tf.Sequential, startSequence: string[], maxDepth: number, vocab: string[], layerToIdx: { [key: string]: number }, idxToLayer: { [key: number]: string }) {
    type QueueItem = { sequence: string[], score: number };
    const priorityQueue: QueueItem[] = []; // Priority queue for tree search
    priorityQueue.push({ sequence: startSequence, score: 1.0 }); // Use 1.0 as the base score

    let bestSequence: QueueItem['sequence'] | null = null;
    let bestScore = 0;

    // While loop avoids recursion to reduce memory footprint
    while (priorityQueue.length > 0) {
        // Dequeue the highest-priority sequence
        const current = priorityQueue.shift()!;
        const { sequence, score } = current;

        // Once we reach out max depth, see if the score is better than the current best score
        if (sequence.length >= maxDepth) {
            if (score > bestScore) {
                bestScore = score;
                bestSequence = sequence;
            }
            continue;
        }

        // Predict probabilities for the next layer
        const inputSeq = sequence.map(layer => layerToIdx[layer]);
        const paddedInput = padSequence(inputSeq, maxDepth);
        const prediction = model.predict(tf.tensor2d([paddedInput])) as tf.Tensor;
        const probs = (await prediction.array())[0] as number[];

        // Expand the tree for top-k predictions
        probs.forEach((prob, idx) => {
            // Each probability greater than 0.1 creates a new path to explore in the tree
            if (prob > 0.1) { // Ignore low-probability paths
                priorityQueue.push({
                    sequence: [...sequence, idxToLayer[idx]],
                    score: score * prob // Calculate the new score as a cumulative probability that punishes low-probability paths
                });
            }
        });

        // Sort the queue by score (descending)
        priorityQueue.sort((a, b) => b.score - a.score);
    }

    return { bestSequence, bestScore };
}

// Define starting sequence and max depth
const startSequence = ['Dataset'];
const maxDepth = 4;

// // Run tree search
const { bestSequence, bestScore } = await treeSearch(
    model,
    startSequence,
    maxDepth,
    vocab,
    layerToIdx,
    idxToLayer
);

console.log('Best Sequence:', bestSequence);
console.log('Best Score:', bestScore);


