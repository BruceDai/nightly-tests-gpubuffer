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

  it('test reshape converted from reshape_weights_as_inputs_relaxed test', async function() {
    // Converted test case (from: V1_1/reshape_weights_as_inputs_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), [1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const op2 = [-1];
    const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const op3 = builder.reshape(op1, op2);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([9]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([9]), outputs.op3.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
