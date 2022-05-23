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

  it('test concat converted from concat_float_1_relaxed test', async function() {
    // Converted test case (from: V1_1/concat_float_1_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 3]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 3]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3]), [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])};
    const axis0 = 0;
    const expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    const result = builder.concat([op1, op2], axis0);
    const graph = builder.build({result});
    const outputs = {result: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 3]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([4, 3]), outputs.result.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
