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

  it('test pad converted from pad_low_rank test', async function() {
    // Converted test case (from: V1_2/pad_low_rank.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [1.0, 2.0, 3.0])};
    const paddings = builder.constant({type: 'int32', dimensions: [1, 2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2]), [3, 1])});
    const expected = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0];
    const output0 = builder.pad(input0, paddings);
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([7]))}};
    graph.compute({'input0': input0Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([7]), outputs.output0.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
