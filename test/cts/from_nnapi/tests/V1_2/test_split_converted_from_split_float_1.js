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

  it('test split converted from split_float_1 test', async function() {
    // Converted test case (from: V1_2/split_float_1.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [6]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([6]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])};
    const axis = 0;
    const numSplits = 3;
    const expected = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    const [output0, output1, output2] = builder.split(input0, numSplits, {'axis': axis});
    const graph = builder.build({output0, output1, output2});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}, output1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}, output2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}};
    graph.compute({'input0': input0Resource}, outputs);
    for (let i = 0; i < 3; i++) {
      utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2]), outputs[['output0', 'output1', 'output2'][i]].resource), expected[i], utils.ctsFp32RestrictAccuracyCriteria);
    }
  });

  it('test split converted from split_float_1_relaxed test', async function() {
    // Converted test case (from: V1_2/split_float_1.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [6]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([6]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])};
    const axis = 0;
    const numSplits = 3;
    const expected = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    const [output0, output1, output2] = builder.split(input0, numSplits, {'axis': axis});
    const graph = builder.build({output0, output1, output2});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}, output1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}, output2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}};
    graph.compute({'input0': input0Resource}, outputs);
    for (let i = 0; i < 3; i++) {
      utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2]), outputs[['output0', 'output1', 'output2'][i]].resource), expected[i], utils.ctsFp32RelaxedAccuracyCriteria);
    }
  });
});
/* eslint-disable max-len */
