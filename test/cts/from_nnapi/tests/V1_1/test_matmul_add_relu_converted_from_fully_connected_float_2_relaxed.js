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

  it('test matmul + add + relu converted from fully_connected_float_2_relaxed test', async function() {
    // Converted test case (from: V1_1/fully_connected_float_2_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 8]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 8]), [0.503691, 0.196961, 0.521017, 0.554248, 0.288678, 0.792476, 0.561653, 0.46223, 0.650736, 0.163132, 0.029658, 0.411544, 0.470539, 0.57239, 0.538755, 0.21203])};
    const op2 = builder.constant({type: 'float32', dimensions: [8, 16]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([16, 8]), [0.091327, 0.063031, 0.109758, 0.266455, 0.096086, 0.142238, -0.104321, -0.252595, -0.304651, -0.128762, -0.339283, 0.103532, -0.139061, -0.241619, 0.265148, 0.302309, 0.103366, 0.19167, 0.008307, 0.051517, 0.131157, 0.08654, -0.176035, -0.16175, -0.070958, -0.03578, 0.333071, -0.034284, -0.048873, 0.357835, 0.141627, -0.041084, -0.316505, -0.062001, -0.062657, -0.123448, 0.031164, -0.139154, -0.208587, -0.136403, 0.054598, 0.117262, 0.180827, 0.093299, 0.067557, 0.135762, 0.02012, 0.146334, -0.08312, -0.061504, -0.060962, 0.322464, 0.100638, 0.174268, -0.001019, 0.008308, 0.147113, 0.017177, 0.287583, -0.145361, 0.139038, -0.306764, 0.083815, -0.061511, 0.149366, -0.275581, -0.049782, 0.043282, -0.312191, -0.073161, -0.162032, 0.00571, -0.139112, 0.263335, 0.06635, 0.054001, 0.324106, -0.125982, -0.124556, -0.232605, -0.196636, 0.059388, -0.106719, -0.173782, -0.080923, 0.080072, 0.080824, 0.0966, -0.072798, -0.176612, -0.197947, 0.25057, 0.227041, 0.091916, -0.100124, 0.281324, -0.123672, -0.118497, -0.319482, -0.190381, -0.101318, 0.006874, -0.025021, 0.289839, -0.163335, 0.262961, -0.114449, 0.15701, 0.037793, 0.266587, -0.048159, 0.145408, 0.0628, -0.079224, -0.10365, 0.002013, -0.116614, 0.229382, 0.07446, 0.218816, -0.167863, -0.093654, -0.236035, -0.14348, -0.225747, 0.030135, 0.181172, -0.221897])});
    const b0 = builder.constant({type: 'float32', dimensions: [16]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([16]), [-0.160594, 0.20577, -0.078307, -0.077984, 0.001937, 0.01586, 0.03681, 0.012346, 0.001028, 0.038551, 0.075415, 0.020804, 0.048478, -0.03227, 0.175688, -0.085662])});
    const expected = [0, 0.0732134, 0, 0, 0, 0.280859, 0, 0.128927, 0, 0.0777251, 0, 0.270268, 0.271435, 0.0173503, 0.335465, 0.235562, 0, 0.0745866, 0, 0.051611, 0, 0.253876, 0, 0.0814873, 0, 0.104104, 0, 0.248529, 0.264194, 0, 0.302973, 0.166252];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.relu(interOut1);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 16]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 16]), outputs.op3.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
