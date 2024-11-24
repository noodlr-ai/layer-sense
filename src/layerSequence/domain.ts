import * as tf from '@tensorflow/tfjs-node';

export type LayerType =
    | 'dense'
    | 'dropout'
    | 'batchNormalization'
    | 'dataset'
    | 'rowSplit'
    | 'columnSplit'
    | 'columnSplice'
    | 'training'
    | 'launcher'
    | 'manualDataEntry'
    | 'dataViewer'
    | 'transformer';

export type LayerTypeWithPad = 'PAD' | LayerType;

// Vocabulary of layers
export const vocab: LayerTypeWithPad[] = ['PAD', 'dataset', 'rowSplit', 'columnSplit', 'columnSplice', 'dense', 'dropout', 'batchNormalization', 'training', 'launcher', 'manualDataEntry', 'dataViewer', 'transformer'];

//  layerToIdx maps a layer to its index in the vocabulary
export function layerToIdx(layer: typeof vocab[0]): number {
    const layerToIdx = Object.fromEntries(vocab.map((layer, idx) => [layer, idx]));
    return layerToIdx[layer];
}

// idxToLayer maps an index to the layer in the vocabulary
export function idxToLayer(idx: number): typeof vocab[0] {
    const idxToLayer = Object.fromEntries(vocab.map((layer, idx) => [idx, layer]));
    return idxToLayer[idx];
}

export function padSequence(sequence: number[], maxLen: number) {
    const pad = new Array(maxLen - sequence.length).fill(layerToIdx('PAD'));
    return [...sequence, ...pad];
}

// padSequences ensures all of the x-values are the same length
export function padSequences(sequences: number[][], maxLen: number) {
    return sequences.map(seq => padSequence(seq, maxLen));
}

// Note: the model input will be the maximum sequences.length - 1
// Turns a sequence of layers into a sequence of indices
export function encodeSequences(sequences: LayerType[][]) {
    return sequences.map(seq => seq.map(layer => layerToIdx(layer)));
}

// This function splits a sequence into multiple input-output pairs for training
// Each sequence will result it in multiple input-output pairs
export function buildTrainingData(encodedSequences: number[][]) {
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

export function convertSequenceIntoTensors(sequence: LayerType[]) {
    const inputSeq = sequence.map(layer => layerToIdx(layer));
    // const paddedSeq = padSequence(inputSeq, maxLen);
    return tf.tensor2d([inputSeq]);
}
