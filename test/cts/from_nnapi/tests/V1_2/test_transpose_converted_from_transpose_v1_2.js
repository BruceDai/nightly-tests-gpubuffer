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

  it('test transpose converted from transpose_v1_2 test', async function() {
    // Converted test case (from: V1_2/transpose_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2]), [1.0, 2.0, 3.0, 4.0])};
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input);
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2]), outputs.output.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test transpose converted from transpose_v1_2_relaxed test', async function() {
    // Converted test case (from: V1_2/transpose_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2]), [1.0, 2.0, 3.0, 4.0])};
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input);
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2]), outputs.output.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
