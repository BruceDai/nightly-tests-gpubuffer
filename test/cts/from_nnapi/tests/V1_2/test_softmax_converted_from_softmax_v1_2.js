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

  it('test softmax converted from softmax_v1_2_axis_dim2_axis1 test', async function() {
    // Converted test case (from: V1_2/softmax_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 5]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]), [17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0])};
    const expected = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];
    const op2 = builder.softmax(op1);
    const graph = builder.build({op2});
    const outputs = {op2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 5]), outputs.op2.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test softmax converted from softmax_v1_2_axis_dim2_axis1_neg test', async function() {
    // Converted test case (from: V1_2/softmax_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 5]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]), [17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0])};
    const expected = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];
    const op2 = builder.softmax(op1);
    const graph = builder.build({op2});
    const outputs = {op2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 5]), outputs.op2.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test softmax converted from softmax_v1_2_axis_relaxed_dim2_axis1 test', async function() {
    // Converted test case (from: V1_2/softmax_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 5]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]), [17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0])};
    const expected = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];
    const op2 = builder.softmax(op1);
    const graph = builder.build({op2});
    const outputs = {op2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 5]), outputs.op2.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test softmax converted from softmax_v1_2_axis_relaxed_dim2_axis1_neg test', async function() {
    // Converted test case (from: V1_2/softmax_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 5]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]), [17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0])};
    const expected = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];
    const op2 = builder.softmax(op1);
    const graph = builder.build({op2});
    const outputs = {op2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 5]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 5]), outputs.op2.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
