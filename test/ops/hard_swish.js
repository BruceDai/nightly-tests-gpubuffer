'use strict';
import * as utils from '../utils.js';

describe('test hardSwish', function() {
  let device;
  let context;
  before(async () => {
    const adaptor = await navigator.gpu.requestAdapter();
    device = await adaptor.requestDevice();
    context = navigator.ml.createContext(device);
  });

  it('hardSwish', async function() {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: [2, 3]});
    const y = builder.hardSwish(x);
    const graph = builder.build({y});
    const inputs = {
      'x': {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3]), [
        -4.2, -3.001, -3., 0.6, 2.994, 3.001,
      ])},
    };
    const outputBuffer = await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3]));
    const outputs = {'y': {resource: outputBuffer}};    
    graph.compute(inputs, outputs);
    const expected = [
      0., 0., 0., 0.36, 2.991006, 3.001,
    ];
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 3]), outputBuffer), expected);
  });
});
