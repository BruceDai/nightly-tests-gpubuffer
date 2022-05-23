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

  it('test clamp converted from relu6_float_1_relaxed test', async function() {
    // Converted test case (from: V1_1/relu6_float_1_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [-10.0, -0.5, 0.5, 10.0])};
    const expected = [0.0, 0.0, 0.5, 6.0];
    const op2 = builder.clamp(op1, {minValue: 0, maxValue: 6});
    const graph = builder.build({op2});
    const outputs = {op2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op2.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
