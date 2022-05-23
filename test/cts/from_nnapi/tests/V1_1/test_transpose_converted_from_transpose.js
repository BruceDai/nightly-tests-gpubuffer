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

  it('test transpose converted from transpose test', async function() {
    // Converted test case (from: V1_1/transpose.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [1.0, 2.0, 3.0, 4.0])};
    const perms = [0, 2, 1, 3];
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input, {'permutation': perms});
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.output.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
