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

  it('test add + clamp converted from add_relaxed test', async function() {
    // Converted test case (from: V1_1/add_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [1.0, 2.0])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [3.0, 4.0])};
    const expected = [4.0, 6.0];
    const interOut0 = builder.add(op1, op2);
    const op3 = builder.clamp(interOut0);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2]), outputs.op3.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
