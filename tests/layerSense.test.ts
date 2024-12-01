import * as tf from '@tensorflow/tfjs-node';
import { LayerType, convertSequenceIntoTensors, getProbabilities, idxToLayer, layerToIdx } from '../src/layerSequence/domain';
import { treeSearch } from '../src/layerSequence/treeSearch';

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