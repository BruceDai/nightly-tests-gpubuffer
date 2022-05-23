'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  let device;
  let context;
  before(async () => {
    const adaptor = await navigator.gpu.requestAdapter();
    device = await adaptor.requestDevice();
    context = navigator.ml.createContext(device);
  });

  it('test softmax converted from softmax_float_2 test', async function() {
    // Converted test case (from: V1_0/softmax_float_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 5]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]), [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0])};
    const expected = [0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647, 0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231];
    const output = builder.softmax(input);
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 5]), outputs.output.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
