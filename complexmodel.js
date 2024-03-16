const width = 1000;
const height = 600;
let hiddenLayersValue2 = 3; // Default number of hidden layers 2
let nodesValue2 = [16, 8, 4]; // Default number of nodes
let isInitialSetup2 = true;
let networkGraph2;
let model2;

import {drawGraph} from './script.js';


class CustomCallback2 extends tf.Callback {
    onBatchEnd(epoch, logs) {
        let weights;
        weights = model2.layers[0].getWeights()[0].arraySync();
        for (let i = 1; i < model2.layers.length; i++) {
            console.log(model2.layers[i])
            weights = weights.concat(model2.layers[i].getWeights()[0].arraySync())
        }
        weights = weights.flat()
        console.log(d3.select("#neuralNet2").selectAll('.link'))
        d3.select("#neuralNet2").selectAll('.link')
        .style("stroke-width", function (d) {
            d.value = weights[d.overall]
            return 4 * d.value ** 2
        })
        .style("stroke", function (d) {
            if (d.value>=0){
                return "green"
            }else{
                return "red"
            }
        })
    }
}

async function trainModel2(model, inputs, labels) {
    let surface = document.getElementById('plot2');
    const batchSize = 8;
    const epochs = 30;
    const callbacks = tfvis.show.fitCallbacks(surface, ['loss'], { callbacks: ['onEpochEnd'] });
    return await model.fit(inputs, labels,
        { batchSize, epochs, shuffle: true, callbacks: [callbacks, new CustomCallback2()] }
    );
}

function createComplexModel() {
    // Adjust the model creation based on the selected number of nodes and hidden layers
    let model2 = tf.sequential();
    model2.add(tf.layers.dense({ inputShape: [13], units: nodesValue2[0], useBias: true, activation: 'relu' }));

    for (let i = 1; i < hiddenLayersValue2; i++) {
        model2.add(tf.layers.dense({ units: nodesValue2[i], activation: 'relu', useBias: true }));
    }

    model2.add(tf.layers.dense({ units: 1, useBias: true }));
    const myOptimizer = tf.train.sgd(.001) 
    model2.compile({ loss: 'meanSquaredError', optimizer: myOptimizer });
    return model2;
}

function draw2() {
    if (isInitialSetup2) {
        let svg = d3.select("#neuralNet2").append("svg")
            .attr("width", width)
            .attr("height", height);
        networkGraph2 = buildNodeGraph2();
        drawGraph(networkGraph2, svg);
        isInitialSetup2 = false;
        svg.selectAll(".link").style("stroke-opacity", .4)
    } else {
        let svg = d3.select("#neuralNet2").select("svg")
        svg.selectAll("*").remove()
        console.log("drawing   " + new Date());
        networkGraph2 = buildNodeGraph2();
        drawGraph(networkGraph2, svg);
    }
    model2 = createComplexModel();
}

function buildNodeGraph2() {
    let newGraph = {
        "nodes": []
    };

    // Construct input layer
    let newFirstLayer2 = [];
    for (let i = 0; i < 13; i++) {
        let newTempLayer2 = { "label": "i" + i, "layer": 1 };
        newFirstLayer2.push(newTempLayer2);
    }

    // Construct hidden layers
    let hiddenLayers2 = [];
    for (let hiddenLayerLoop = 0; hiddenLayerLoop < hiddenLayersValue2; hiddenLayerLoop++) {
        let newHiddenLayer = [];
        // For the height of this hidden layer
        for (let i = 0; i < nodesValue2[hiddenLayerLoop]; i++) {
            let newTempLayer = { "label": "h" + hiddenLayerLoop + i, "layer": (hiddenLayerLoop + 2) };
            newHiddenLayer.push(newTempLayer);
        }
        hiddenLayers2.push(newHiddenLayer);
    }

    // Construct output layer
    let newOutputLayer = [];
    for (let i = 0; i < 1; i++) {
        let newTempLayer = { "label": "o" + i, "layer": hiddenLayersValue2 + 2 };
        newOutputLayer.push(newTempLayer);
    }

    // Add to newGraph
    let allMiddle = newGraph.nodes.concat.apply([], hiddenLayers2);
    newGraph.nodes = newGraph.nodes.concat(newFirstLayer2, allMiddle, newOutputLayer);

    console.log(newGraph);
    return newGraph;
}

export {CustomCallback2, trainModel2, createComplexModel, draw2, buildNodeGraph2, hiddenLayersValue2, nodesValue2, networkGraph2, model2};