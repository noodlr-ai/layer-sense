import * as tf from '@tensorflow/tfjs-node';
import { LayerType, convertSequenceIntoTensors, idxToLayer, layerToIdx } from '../src/layerSequence/domain';

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

function getProbabilities(pTensor: tf.Tensor | tf.Tensor[]) {
    const probs: number[] = Array.isArray(pTensor) ? pTensor[0].squeeze().arraySync() as number[] : pTensor.squeeze().arraySync() as number[];
    return probs.map((prob, idx) => ({ layer: idxToLayer(idx), prob })).filter(d => d.layer !== 'PAD').sort((a, b) => b.prob - a.prob);
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
        expect(probs[0].layer).toEqual('transformer');
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
});