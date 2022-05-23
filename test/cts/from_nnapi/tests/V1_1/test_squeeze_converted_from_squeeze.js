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

  it('test squeeze converted from squeeze test', async function() {
    // Converted test case (from: V1_1/squeeze.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [4, 1, 1, 2]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 1, 1, 2]), [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1])};
    const squeezeDims = [1, 2];
    const expected = [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1];
    const output = builder.squeeze(input, {'axes': squeezeDims});
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([4, 2]), outputs.output.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
