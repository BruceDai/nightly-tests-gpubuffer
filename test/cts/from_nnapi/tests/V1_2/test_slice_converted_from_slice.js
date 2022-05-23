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

  it('test slice converted from slice test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [4]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [1, 2, 3, 4])};
    const begin = [1];
    const size = [2];
    const expected = [2, 3];
    const output = builder.slice(input, begin, size);
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2]), outputs.output.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [4]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [1, 2, 3, 4])};
    const begin = [1];
    const size = [2];
    const expected = [2, 3];
    const output = builder.slice(input, begin, size);
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2]), outputs.output.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_2 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 3]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3]), [1, 2, 3, 4, 5, 6])};
    const begin1 = [1, 0];
    const size1 = [1, 2];
    const expected = [4, 5];
    const output1 = builder.slice(input1, begin1, size1);
    const graph = builder.build({output1});
    const outputs = {output1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2]))}};
    graph.compute({'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2]), outputs.output1.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 3]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3]), [1, 2, 3, 4, 5, 6])};
    const begin1 = [1, 0];
    const size1 = [1, 2];
    const expected = [4, 5];
    const output1 = builder.slice(input1, begin1, size1);
    const graph = builder.build({output1});
    const outputs = {output1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2]))}};
    graph.compute({'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2]), outputs.output1.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_3 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input2 = builder.input('input2', {type: 'float32', dimensions: [2, 3, 2]});
    const input2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3, 2]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])};
    const begin2 = [0, 0, 0];
    const size2 = [2, 3, 2];
    const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const output2 = builder.slice(input2, begin2, size2);
    const graph = builder.build({output2});
    const outputs = {output2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3, 2]))}};
    graph.compute({'input2': input2Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 3, 2]), outputs.output2.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_3 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input2 = builder.input('input2', {type: 'float32', dimensions: [2, 3, 2]});
    const input2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3, 2]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])};
    const begin2 = [0, 0, 0];
    const size2 = [2, 3, 2];
    const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const output2 = builder.slice(input2, begin2, size2);
    const graph = builder.build({output2});
    const outputs = {output2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3, 2]))}};
    graph.compute({'input2': input2Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 3, 2]), outputs.output2.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_4 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input3 = builder.input('input3', {type: 'float32', dimensions: [4, 1, 1, 1]});
    const input3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 1, 1, 1]), [1, 2, 3, 4])};
    const begin3 = [1, 0, 0, 0];
    const size3 = [3, 1, 1, 1];
    const expected = [2, 3, 4];
    const output3 = builder.slice(input3, begin3, size3);
    const graph = builder.build({output3});
    const outputs = {output3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 1]))}};
    graph.compute({'input3': input3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 1]), outputs.output3.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_4 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input3 = builder.input('input3', {type: 'float32', dimensions: [4, 1, 1, 1]});
    const input3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 1, 1, 1]), [1, 2, 3, 4])};
    const begin3 = [1, 0, 0, 0];
    const size3 = [3, 1, 1, 1];
    const expected = [2, 3, 4];
    const output3 = builder.slice(input3, begin3, size3);
    const graph = builder.build({output3});
    const outputs = {output3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 1]))}};
    graph.compute({'input3': input3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 1]), outputs.output3.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_5 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input4 = builder.input('input4', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input4Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin4 = [1, 0, 0, 0];
    const size4 = [1, 1, 3, 1];
    const expected = [3, 3, 3];
    const output4 = builder.slice(input4, begin4, size4);
    const graph = builder.build({output4});
    const outputs = {output4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 1]))}};
    graph.compute({'input4': input4Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 1]), outputs.output4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_5 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input4 = builder.input('input4', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input4Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin4 = [1, 0, 0, 0];
    const size4 = [1, 1, 3, 1];
    const expected = [3, 3, 3];
    const output4 = builder.slice(input4, begin4, size4);
    const graph = builder.build({output4});
    const outputs = {output4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 1]))}};
    graph.compute({'input4': input4Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 1]), outputs.output4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_float16_5 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input4 = builder.input('input4', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input4Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin4 = [1, 0, 0, 0];
    const size4 = [1, 1, 3, 1];
    const expected = [3, 3, 3];
    const output4 = builder.slice(input4, begin4, size4);
    const graph = builder.build({output4});
    const outputs = {output4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 1]))}};
    graph.compute({'input4': input4Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 1]), outputs.output4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_6 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input5 = builder.input('input5', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input5Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin5 = [1, 0, 0, 0];
    const size5 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output5 = builder.slice(input5, begin5, size5);
    const graph = builder.build({output5});
    const outputs = {output5: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]))}};
    graph.compute({'input5': input5Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]), outputs.output5.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_6 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input5 = builder.input('input5', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input5Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin5 = [1, 0, 0, 0];
    const size5 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output5 = builder.slice(input5, begin5, size5);
    const graph = builder.build({output5});
    const outputs = {output5: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]))}};
    graph.compute({'input5': input5Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]), outputs.output5.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_float16_6 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input5 = builder.input('input5', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input5Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin5 = [1, 0, 0, 0];
    const size5 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output5 = builder.slice(input5, begin5, size5);
    const graph = builder.build({output5});
    const outputs = {output5: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]))}};
    graph.compute({'input5': input5Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]), outputs.output5.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_8 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input7 = builder.input('input7', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input7Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin7 = [1, 0, 0, 0];
    const size7 = [2, 1, -1, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output7 = builder.slice(input7, begin7, size7);
    const graph = builder.build({output7});
    const outputs = {output7: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]))}};
    graph.compute({'input7': input7Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]), outputs.output7.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_8 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input7 = builder.input('input7', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input7Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin7 = [1, 0, 0, 0];
    const size7 = [2, 1, -1, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output7 = builder.slice(input7, begin7, size7);
    const graph = builder.build({output7});
    const outputs = {output7: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]))}};
    graph.compute({'input7': input7Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]), outputs.output7.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_float16_8 test', async function() {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input7 = builder.input('input7', {type: 'int32', dimensions: [3, 2, 3, 1]});
    const input7Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 3, 1]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])};
    const begin7 = [1, 0, 0, 0];
    const size7 = [2, 1, -1, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output7 = builder.slice(input7, begin7, size7);
    const graph = builder.build({output7});
    const outputs = {output7: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]))}};
    graph.compute({'input7': input7Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1, 3, 1]), outputs.output7.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
