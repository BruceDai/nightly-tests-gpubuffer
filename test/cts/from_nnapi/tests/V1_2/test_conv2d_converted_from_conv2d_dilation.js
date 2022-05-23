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

  it('test conv2d (fused ops) converted from conv2d_dilation_nhwc test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])});
    const op3 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])});
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nhwc';
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])});
    const op3 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])});
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nhwc';
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])};
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1]});
    const op3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nhwc';
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])};
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1]});
    const op3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nhwc';
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nchw test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])});
    const op3 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])});
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nchw';
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])});
    const op3 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])});
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nchw';
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])};
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1]});
    const op3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nchw';
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])};
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1]});
    const op3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nchw';
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nhwc_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 9, 9, 1]});
    const op11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 9, 9, 1]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])};
    const op21 = builder.constant({type: 'float32', dimensions: [1, 3, 3, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1, 2, 3, 4, 5, 6, 7, 8, 9])});
    const op31 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])});
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nhwc';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const op41 = builder.conv2d(op11, op21, {'bias': op31, 'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'ohwi'});
    const graph = builder.build({op41});
    const outputs = {op41: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]))}};
    graph.compute({'op11': op11Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), outputs.op41.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nhwc_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 9, 9, 1]});
    const op11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 9, 9, 1]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])};
    const op21 = builder.constant({type: 'float32', dimensions: [1, 3, 3, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1, 2, 3, 4, 5, 6, 7, 8, 9])});
    const op31 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])});
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nhwc';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const op41 = builder.conv2d(op11, op21, {'bias': op31, 'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'ohwi'});
    const graph = builder.build({op41});
    const outputs = {op41: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]))}};
    graph.compute({'op11': op11Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), outputs.op41.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nhwc_weight_as_input_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 9, 9, 1]});
    const op11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 9, 9, 1]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])};
    const op21 = builder.input('op21', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op21Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const op31 = builder.input('op31', {type: 'float32', dimensions: [1]});
    const op31Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nhwc';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const op41 = builder.conv2d(op11, op21, {'bias': op31, 'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'ohwi'});
    const graph = builder.build({op41});
    const outputs = {op41: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]))}};
    graph.compute({'op11': op11Resource, 'op21': op21Resource, 'op31': op31Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), outputs.op41.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nhwc_weight_as_input_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 9, 9, 1]});
    const op11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 9, 9, 1]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])};
    const op21 = builder.input('op21', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op21Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const op31 = builder.input('op31', {type: 'float32', dimensions: [1]});
    const op31Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nhwc';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const op41 = builder.conv2d(op11, op21, {'bias': op31, 'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'ohwi'});
    const graph = builder.build({op41});
    const outputs = {op41: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]))}};
    graph.compute({'op11': op11Resource, 'op21': op21Resource, 'op31': op31Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), outputs.op41.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nchw_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 1, 9, 9]});
    const op11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 9, 9]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])};
    const op21 = builder.constant({type: 'float32', dimensions: [1, 3, 3, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1, 2, 3, 4, 5, 6, 7, 8, 9])});
    const op31 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])});
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nchw';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const op41 = builder.conv2d(op11, op21, {'bias': op31, 'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'ohwi'});
    const graph = builder.build({op41});
    const outputs = {op41: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]))}};
    graph.compute({'op11': op11Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), outputs.op41.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nchw_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 1, 9, 9]});
    const op11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 9, 9]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])};
    const op21 = builder.constant({type: 'float32', dimensions: [1, 3, 3, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1, 2, 3, 4, 5, 6, 7, 8, 9])});
    const op31 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])});
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nchw';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const op41 = builder.conv2d(op11, op21, {'bias': op31, 'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'ohwi'});
    const graph = builder.build({op41});
    const outputs = {op41: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]))}};
    graph.compute({'op11': op11Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), outputs.op41.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nchw_weight_as_input_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 1, 9, 9]});
    const op11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 9, 9]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])};
    const op21 = builder.input('op21', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op21Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const op31 = builder.input('op31', {type: 'float32', dimensions: [1]});
    const op31Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nchw';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const op41 = builder.conv2d(op11, op21, {'bias': op31, 'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'ohwi'});
    const graph = builder.build({op41});
    const outputs = {op41: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]))}};
    graph.compute({'op11': op11Resource, 'op21': op21Resource, 'op31': op31Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), outputs.op41.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_dilation_nchw_weight_as_input_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = new MLGraphBuilder(context);
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 1, 9, 9]});
    const op11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 9, 9]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])};
    const op21 = builder.input('op21', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op21Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const op31 = builder.input('op31', {type: 'float32', dimensions: [1]});
    const op31Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nchw';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const op41 = builder.conv2d(op11, op21, {'bias': op31, 'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'ohwi'});
    const graph = builder.build({op41});
    const outputs = {op41: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]))}};
    graph.compute({'op11': op11Resource, 'op21': op21Resource, 'op31': op31Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 3, 3]), outputs.op41.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
