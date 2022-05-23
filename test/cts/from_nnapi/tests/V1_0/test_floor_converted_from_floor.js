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

  it('test floor converted from floor test', async function() {
    // Converted test case (from: V1_0/floor.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 10.2])};
    const expected = [-2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 10];
    const op2 = builder.floor(op1);
    const graph = builder.build({op2});
    const outputs = {op2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), outputs.op2.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
