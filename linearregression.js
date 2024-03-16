let isInitialSetup3 = true;
const width = 1000;
const height = 600;
let networkGraph3;
let linearModel, linearModel2;


import {drawGraph2, samples} from './script.js';
export {buildNodeGraph3, draw3, trainLinearModel, networkGraph3, linearModel, linearModel2, createLinear, trainLinearModel2};
import {generatePredictionMatrix, predictZones, updateBasicHeatmap, updateBasicHeatmap2} from './heatmap.js';


function buildNodeGraph3() {
    let newGraph = {
        "nodes": []
    };

    // Construct input layer
    let newFirstLayer2 = [];
    for (let i = 0; i < 2; i++) {
        let newTempLayer2 = { "label": "i" + i, "layer": 1 };
        newFirstLayer2.push(newTempLayer2);
    }

    // Construct hidden layers

    // Construct output layer
    let newOutputLayer = [];
    for (let i = 0; i < 1; i++) {
        let newTempLayer = { "label": "o" + i, "layer": 2 };
        newOutputLayer.push(newTempLayer);
    }

    // Add to newGraph
    newGraph.nodes = newGraph.nodes.concat(newFirstLayer2, newOutputLayer);

    console.log(newGraph);
    return newGraph;
}

function draw3() {
    if (isInitialSetup3) {
        let svg = d3.select("#neuralNet3").append("svg")
            .attr("width", width)
            .attr("height", height);
        networkGraph3 = buildNodeGraph3();
        drawGraph2(networkGraph3, svg);
        isInitialSetup3 = false;
        svg.selectAll(".link").style("stroke-opacity", .4)
    } else {
        let svg = d3.select("#neuralNet3").select("svg")
        svg.selectAll("*").remove()
        console.log("drawing   " + new Date());
        networkGraph3 = buildNodeGraph3();
        drawGraph2(networkGraph3, svg);
    }
    linearModel = createLinear();
    linearModel2 = createLinear();
}

function createLinear() {
    // Adjust the model creation based on the selected number of nodes and hidden layers
    let model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [2], units: 1, useBias: true, activation: 'sigmoid' }));
    const myOptimizer = tf.train.adam(.03) 
    model.compile({ loss: 'binaryCrossentropy', optimizer: myOptimizer });
    return model;
}


// Add the CustomCallback class
class CustomCallback extends tf.Callback {
    onBatchEnd(epoch, logs) {
        let weights = linearModel.layers[0].getWeights()[0].arraySync();
        for (let i = 1; i < linearModel.layers.length; i++) {
            weights = weights.concat(model.layers[i].getWeights()[0].arraySync())
        }
        weights = weights.flat()
        d3.select("#neuralNet3").selectAll('.link')
        .style("stroke-width", function (d) {
            d.value = weights[d.overall]
            return d.value ** 2
        })
        .style("stroke", function (d) {
            if (d.value>=0){
                return "green"
            }else{
                return "red"
            }
        })
    }

    async onEpochEnd(epoch, logs) {
        let preds = await predictZones(samples, linearModel)
        await updateBasicHeatmap(preds)
    }
}


async function trainLinearModel(model, inputs, labels) {
    let surface = document.getElementById('lossPlot1');
    const batchSize = 64;
    const epochs = 20;
    const callbacks = tfvis.show.fitCallbacks(surface, ['loss'], { callbacks: ['onEpochEnd'] });
    return await model.fit(inputs, labels,
        { batchSize, epochs, shuffle: true, callbacks:  [callbacks, new CustomCallback()] }
    );
}


// Add the CustomCallback class for the 2nd viz
class CustomCallback2 extends tf.Callback {
    onBatchEnd(epoch, logs) {
        let weights = linearModel2.layers[0].getWeights()[0].arraySync();
        for (let i = 1; i < linearModel.layers.length; i++) {
            weights = weights.concat(model.layers[i].getWeights()[0].arraySync())
        }
        weights = weights.flat()
        d3.select("#neuralNet3").selectAll('.link')
        .style("stroke-width", function (d) {
            d.value = weights[d.overall]
            return d.value ** 2
        })
        .style("stroke", function (d) {
            if (d.value>=0){
                return "green"
            }else{
                return "red"
            }
        })
    }

    async onEpochEnd(epoch, logs) {
        let preds = await predictZones(samples, linearModel2)
        await updateBasicHeatmap2(preds)
    }
}


async function trainLinearModel2(model, inputs, labels) {
    let surface = document.getElementById('lossPlot2');
    const batchSize = 128;
    const epochs = 50;
    const callbacks = tfvis.show.fitCallbacks(surface, ['loss'], { callbacks: ['onEpochEnd'] });
    return await model.fit(inputs, labels,
        { batchSize, epochs, shuffle: true, callbacks: [callbacks, new CustomCallback2()] }
    );
}
