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

  it('test pad converted from pad test', async function() {
    // Converted test case (from: V1_1/pad.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [1.0, 2.0, 3.0, 4.0])};
    const op2 = builder.constant({type: 'int32', dimensions: [4, 2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 2]), [0, 0, 1, 1, 1, 1, 0, 0])});
    const expected = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    const op3 = builder.pad(op1, op2);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 4, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 4, 4, 1]), outputs.op3.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
