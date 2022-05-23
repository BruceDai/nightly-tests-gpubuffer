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

  it('test conv2d (fused ops) converted from conv2d_v1_2_nhwc test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
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
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
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
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
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
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
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
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_nchw test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
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
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
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
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
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
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
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
    const expected = [0.875, 0.875, 0.875, 0.875];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 2, 2]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_channel_nhwc test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), [5.0, 5.0, 5.0])};
    const op22 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])});
    const op32 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])});
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nhwc';
    const expected = [15.0, 37.5, 60.0];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]))}};
    graph.compute({'op12': op12Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), outputs.op42.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_channel_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), [5.0, 5.0, 5.0])};
    const op22 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])});
    const op32 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])});
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nhwc';
    const expected = [15.0, 37.5, 60.0];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]))}};
    graph.compute({'op12': op12Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), outputs.op42.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_channel_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), [5.0, 5.0, 5.0])};
    const op22 = builder.input('op22', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op22Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])};
    const op32 = builder.input('op32', {type: 'float32', dimensions: [3]});
    const op32Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nhwc';
    const expected = [15.0, 37.5, 60.0];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]))}};
    graph.compute({'op12': op12Resource, 'op22': op22Resource, 'op32': op32Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), outputs.op42.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_channel_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), [5.0, 5.0, 5.0])};
    const op22 = builder.input('op22', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op22Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])};
    const op32 = builder.input('op32', {type: 'float32', dimensions: [3]});
    const op32Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nhwc';
    const expected = [15.0, 37.5, 60.0];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]))}};
    graph.compute({'op12': op12Resource, 'op22': op22Resource, 'op32': op32Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), outputs.op42.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_channel_nchw test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]), [5.0, 5.0, 5.0])};
    const op22 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])});
    const op32 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])});
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nchw';
    const expected = [15.0, 37.5, 60.0];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]))}};
    graph.compute({'op12': op12Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]), outputs.op42.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_channel_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]), [5.0, 5.0, 5.0])};
    const op22 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])});
    const op32 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])});
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nchw';
    const expected = [15.0, 37.5, 60.0];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]))}};
    graph.compute({'op12': op12Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]), outputs.op42.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_channel_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]), [5.0, 5.0, 5.0])};
    const op22 = builder.input('op22', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op22Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])};
    const op32 = builder.input('op32', {type: 'float32', dimensions: [3]});
    const op32Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nchw';
    const expected = [15.0, 37.5, 60.0];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]))}};
    graph.compute({'op12': op12Resource, 'op22': op22Resource, 'op32': op32Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]), outputs.op42.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_channel_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]), [5.0, 5.0, 5.0])};
    const op22 = builder.input('op22', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op22Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])};
    const op32 = builder.input('op32', {type: 'float32', dimensions: [3]});
    const op32Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nchw';
    const expected = [15.0, 37.5, 60.0];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]))}};
    graph.compute({'op12': op12Resource, 'op22': op22Resource, 'op32': op32Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 1, 1]), outputs.op42.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_large_nhwc test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])};
    const op23 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])});
    const op33 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])});
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nhwc';
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]))}};
    graph.compute({'op13': op13Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]), outputs.op43.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_large_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])};
    const op23 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])});
    const op33 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])});
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nhwc';
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]))}};
    graph.compute({'op13': op13Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]), outputs.op43.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_large_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])};
    const op23 = builder.input('op23', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op23Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])};
    const op33 = builder.input('op33', {type: 'float32', dimensions: [3]});
    const op33Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nhwc';
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]))}};
    graph.compute({'op13': op13Resource, 'op23': op23Resource, 'op33': op33Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]), outputs.op43.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_large_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])};
    const op23 = builder.input('op23', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op23Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])};
    const op33 = builder.input('op33', {type: 'float32', dimensions: [3]});
    const op33Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nhwc';
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]))}};
    graph.compute({'op13': op13Resource, 'op23': op23Resource, 'op33': op33Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 3, 3]), outputs.op43.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_large_nchw test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 3, 2, 3]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]), [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])};
    const op23 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])});
    const op33 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])});
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nchw';
    const expected = [30.0, 66.0, 102.0, 138.0, 174.0, 210.0, 36.0, 81.0, 126.0, 171.0, 216.0, 261.0, 42.0, 96.0, 150.0, 204.0, 258.0, 312.0];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]))}};
    graph.compute({'op13': op13Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]), outputs.op43.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_large_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 3, 2, 3]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]), [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])};
    const op23 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])});
    const op33 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])});
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nchw';
    const expected = [30.0, 66.0, 102.0, 138.0, 174.0, 210.0, 36.0, 81.0, 126.0, 171.0, 216.0, 261.0, 42.0, 96.0, 150.0, 204.0, 258.0, 312.0];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]))}};
    graph.compute({'op13': op13Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]), outputs.op43.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_large_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 3, 2, 3]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]), [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])};
    const op23 = builder.input('op23', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op23Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])};
    const op33 = builder.input('op33', {type: 'float32', dimensions: [3]});
    const op33Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nchw';
    const expected = [30.0, 66.0, 102.0, 138.0, 174.0, 210.0, 36.0, 81.0, 126.0, 171.0, 216.0, 261.0, 42.0, 96.0, 150.0, 204.0, 258.0, 312.0];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]))}};
    graph.compute({'op13': op13Resource, 'op23': op23Resource, 'op33': op33Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]), outputs.op43.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from conv2d_v1_2_large_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 3, 2, 3]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]), [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])};
    const op23 = builder.input('op23', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op23Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])};
    const op33 = builder.input('op33', {type: 'float32', dimensions: [3]});
    const op33Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nchw';
    const expected = [30.0, 66.0, 102.0, 138.0, 174.0, 210.0, 36.0, 81.0, 126.0, 171.0, 216.0, 261.0, 42.0, 96.0, 150.0, 204.0, 258.0, 312.0];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'inputLayout': layout, 'filterLayout': 'ohwi'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]))}};
    graph.compute({'op13': op13Resource, 'op23': op23Resource, 'op33': op33Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 2, 3]), outputs.op43.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
