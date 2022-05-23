'use strict';
import * as utils from '../utils.js';

describe('test squeeze', function() {
  let device;
  let context;
  before(async () => {
    const adaptor = await navigator.gpu.requestAdapter();
    device = await adaptor.requestDevice();
    context = navigator.ml.createContext(device);
  });

  async function testSqueeze(oldShape, axes, expectedShape) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: oldShape});
    const y = builder.squeeze(x, {axes});
    const graph = builder.build({y});
    const bufferSize = utils.sizeOfShape(oldShape);
    const inputBuffer = new Float32Array(bufferSize);
    for (let i = 0; i < inputBuffer.length; ++i) {
      inputBuffer[i] = Math.random();
    }
    const inputs = {'x': {resource: await utils.createGPUBuffer(device, bufferSize, inputBuffer)}};
    const outputs = {'y': {resource: await utils.createGPUBuffer(device, utils.sizeOfShape(expectedShape))}};
    graph.compute(inputs, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape(expectedShape), outputs.y.resource), inputBuffer);
  }

  it('squeeze one dimension by default', async function() {
    await testSqueeze([1, 3, 4, 5], undefined, [3, 4, 5]);
  });

  it('squeeze one dimension with axes', async function() {
    await testSqueeze([1, 3, 1, 5], [0], [3, 1, 5]);
  });

  it('squeeze two dimensions by default', async function() {
    await testSqueeze([1, 3, 1, 5], undefined, [3, 5]);
  });

  it('squeeze two dimensions with axes', async function() {
    await testSqueeze([1, 3, 1, 5], [0, 2], [3, 5]);
  });
});
