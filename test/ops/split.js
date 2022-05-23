'use strict';
import * as utils from '../utils.js';

describe('test split', function() {
  let device;
  let context;
  before(async () => {
    const adaptor = await navigator.gpu.requestAdapter();
    device = await adaptor.requestDevice();
    context = navigator.ml.createContext(device);
  });

  async function testSplit(
      inputShape, inputValue, expectedArray, splits, axis = undefined) {
    const builder = new MLGraphBuilder(context);
    const input =
        builder.input('input', {type: 'float32', dimensions: inputShape});
    const splittedOperands = builder.split(input, splits, {axis});
    const namedOperands = {};
    for (let i = 0; i < splittedOperands.length; ++i) {
      namedOperands[`split${i}`] = splittedOperands[i];
    }
    const graph = builder.build(namedOperands);
    const inputs = {'input': {resource: await utils.createGPUBuffer(device, utils.sizeOfShape(inputShape), inputValue)}};
    const outputs = {};
    for (let i = 0; i < splittedOperands.length; ++i) {
      outputs[`split${i}`] =
      {resource: await utils.createGPUBuffer(device, utils.sizeOfShape(expectedArray[i].shape))};
    }
    graph.compute(inputs, outputs);
    for (let i = 0; i < splittedOperands.length; ++i) {
      utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape(expectedArray[i].shape), outputs[`split${i}`].resource), expectedArray[i].value);
    }
  }

  it('split', async function() {
    await testSplit(
        [6], [1, 2, 3, 4, 5, 6],
        [
          {shape: [2], value: [1, 2]},
          {shape: [2], value: [3, 4]},
          {shape: [2], value: [5, 6]},
        ],
        3);

    await testSplit(
        [6], [1, 2, 3, 4, 5, 6],
        [{shape: [2], value: [1, 2]}, {shape: [4], value: [3, 4, 5, 6]}],
        [2, 4]);
  });

  it('split 2d', async function() {
    await testSplit(
        [2, 6], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [
          {shape: [2, 3], value: [1, 2, 3, 7, 8, 9]},
          {shape: [2, 3], value: [4, 5, 6, 10, 11, 12]},
        ],
        2, 1);
    await testSplit(
        [2, 6], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [
          {shape: [2, 2], value: [1, 2, 7, 8]},
          {shape: [2, 4], value: [3, 4, 5, 6, 9, 10, 11, 12]},
        ],
        [2, 4], 1);
  });
});
