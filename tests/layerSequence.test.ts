import * as tf from '@tensorflow/tfjs-node';
import { LayerType, convertSequenceIntoTensors, idxToLayer, layerToIdx } from '../src/layerSequence/domain';
import { treeSearch } from '../src/layerSequence/treeSearch';

export type LayerSenseProbabilities = {
    layer: LayerType;
    prob: number;
}

// displayProbabilities displays the probabilities of the next layer
function displayProbabilities(probs: tf.Tensor | tf.Tensor[]) {
    if (Array.isArray(probs)) {
        const p = probs[0].squeeze().arraySync();
        console.log(p);

    } else {
        const p = probs.squeeze().arraySync() as number[];
        // We add 1 to skip the padding index
        console.log(p.map((prob, idx) => `${idxToLayer(idx)}: ${prob.toFixed(3)}`).join('\n'));
    }
}

// getProbabilities converts tensors into a list of probabilities sorted from highest to lowest
function getProbabilities(pTensor: tf.Tensor | tf.Tensor[]) {
    const probs: number[] = Array.isArray(pTensor) ? pTensor[0].squeeze().arraySync() as number[] : pTensor.squeeze().arraySync() as number[];
    return probs.map((prob, idx) => ({ layer: idxToLayer(idx), prob })).filter((d): d is LayerSenseProbabilities => d.layer !== 'PAD').sort((a, b) => b.prob - a.prob);
}

describe('Test Layer Sequence Model', () => {
    let model: tf.LayersModel;
    it('Loads the model', async () => {
        model = await tf.loadLayersModel('file://build/model/model.json');
        expect(model).toBeDefined
    });

    it('Predicts that the next layer is a transformer', async () => {
        const testSeq: LayerType[] = ['launcher', 'manualDataEntry'];
        const inputSeq = convertSequenceIntoTensors(testSeq);

        const pTensor = model.predict(inputSeq);
        const probs = getProbabilities(pTensor);
        expect(probs[0].layer).toEqual('transformerFillMask');
    });

    it('Predicts that the next layer is a rowSplit', async () => {
        const testSeq: LayerType[] = ['dataset'];
        const inputSeq = convertSequenceIntoTensors(testSeq);

        const pTensor = model.predict(inputSeq);
        const probs = getProbabilities(pTensor);
        expect(probs[0].layer).toEqual('rowSplit');
    });

    it('Predicts that the next layer is a training', async () => {
        const testSeq: LayerType[] = ['dataset', 'rowSplit', 'columnSplit'];
        const inputSeq = convertSequenceIntoTensors(testSeq);

        const pTensor = model.predict(inputSeq);
        const probs = getProbabilities(pTensor);
        expect(probs[0].layer).toEqual('dense');
    });

    it('Predicts that the next layer is a training', async () => {
        const testSeq: LayerType[] = ['dataset', 'rowSplit', 'columnSplit', 'dense', 'dense'];
        const inputSeq = convertSequenceIntoTensors(testSeq);

        const pTensor = model.predict(inputSeq);
        const probs = getProbabilities(pTensor);
        expect(probs[0].layer).toEqual('dense');
    });

    it('Tree search predicts the next two layers to be columnSplit and training', async () => {
        const inputSeq: LayerType[] = ['dataset', 'rowSplit'];
        const seq = await treeSearch(model, inputSeq, 4); // 4 is the max depth
        console.log(seq.bestSequence);
        expect(seq.bestSequence[2]).toEqual('columnSplit');
        expect(seq.bestSequence[3]).toEqual('dense');
    });
});