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

  it('test sub + clamp converted from sub_v1_2_none test', async function() {
    // Converted test case (from: V1_2/sub_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [2.0, -4.0, 8.0, -16.0])};
    const input1 = builder.input('input1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [2.0, -2.0, -4.0, 4.0])};
    const expected = [0.0, -2.0, 12.0, -20.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0);
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'input0': input0Resource, 'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.output0.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_relu test', async function() {
    // Converted test case (from: V1_2/sub_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [2.0, -4.0, 8.0, -16.0])};
    const input1 = builder.input('input1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [2.0, -2.0, -4.0, 4.0])};
    const expected = [0.0, 0.0, 12.0, 0.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.relu(interOut0);
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'input0': input0Resource, 'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.output0.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_relu1 test', async function() {
    // Converted test case (from: V1_2/sub_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [2.0, -4.0, 8.0, -16.0])};
    const input1 = builder.input('input1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [2.0, -2.0, -4.0, 4.0])};
    const expected = [0.0, -1.0, 1.0, -1.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0, {minValue: -1, maxValue: 1});
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'input0': input0Resource, 'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.output0.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_relu6 test', async function() {
    // Converted test case (from: V1_2/sub_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [2.0, -4.0, 8.0, -16.0])};
    const input1 = builder.input('input1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [2.0, -2.0, -4.0, 4.0])};
    const expected = [0.0, 0.0, 6.0, 0.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0, {minValue: 0, maxValue: 6});
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'input0': input0Resource, 'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.output0.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
