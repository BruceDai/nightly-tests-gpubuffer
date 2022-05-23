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

  it('test split converted from split_int32_4 test', async function() {
    // Converted test case (from: V1_2/split_int32_4.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'int32', dimensions: [2, 2, 2]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2]), [1, 2, 3, 4, 5, 6, 7, 8])};
    const axis = 1;
    const numSplits = 2;
    const expected = [[1, 2, 5, 6], [3, 4, 7, 8]];
    const [output0, output1] = builder.split(input0, numSplits, {'axis': axis});
    const graph = builder.build({output0, output1});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 2]))}, output1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 2]))}};
    graph.compute({'input0': input0Resource}, outputs);
    for (let i = 0; i < 2; i++) {
      utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1, 2]), outputs[['output0', 'output1'][i]].resource), expected[i], utils.ctsFp32RestrictAccuracyCriteria);
    }
  });

  it('test split converted from split_int32_4_relaxed test', async function() {
    // Converted test case (from: V1_2/split_int32_4.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'int32', dimensions: [2, 2, 2]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2]), [1, 2, 3, 4, 5, 6, 7, 8])};
    const axis = 1;
    const numSplits = 2;
    const expected = [[1, 2, 5, 6], [3, 4, 7, 8]];
    const [output0, output1] = builder.split(input0, numSplits, {'axis': axis});
    const graph = builder.build({output0, output1});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 2]))}, output1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 2]))}};
    graph.compute({'input0': input0Resource}, outputs);
    for (let i = 0; i < 2; i++) {
      utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1, 2]), outputs[['output0', 'output1'][i]].resource), expected[i], utils.ctsFp32RelaxedAccuracyCriteria);
    }
  });
});
/* eslint-disable max-len */
