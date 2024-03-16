let nodesValue = [4, 2]; // Default number of nodes
let hiddenLayersValue = 2; // Default number of hidden layers
import {model, samples, nmInputs} from './script.js';
import {generatePredictionMatrix, predictZones, updateHeatmap} from './heatmap.js';


// Add the CustomCallback class
class CustomCallback extends tf.Callback {
    onBatchEnd(epoch, logs) {
        let weights = model.layers[0].getWeights()[0].arraySync();
        for (let i = 1; i < model.layers.length; i++) {
            weights = weights.concat(model.layers[i].getWeights()[0].arraySync())
        }
        weights = weights.flat()
        d3.select("#neuralNet").selectAll('.link')
        .style("stroke-width", function (d) {
            d.value = weights[d.overall]
            return d.value*2
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
        let preds = await predictZones(samples, model)
        await updateHeatmap(preds)
    }
}


async function trainModel(model, inputs, labels) {
    let surface = document.getElementById('plot1');
    const batchSize = 64;
    const epochs = 50;
    const callbacks = tfvis.show.fitCallbacks(surface, ['loss'], { callbacks: ['onEpochEnd'] });
    console.log(model.summary())
    return await model.fit(inputs, labels,
        { batchSize, epochs, shuffle: true, callbacks: [callbacks, new CustomCallback()] }
    );
}

function createModel() {
    // Adjust the model creation based on the selected number of nodes and hidden layers
    let model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [2], units: nodesValue[0], useBias: true, activation: 'relu' }));

    for (let i = 1; i < hiddenLayersValue; i++) {
        model.add(tf.layers.dense({ units: nodesValue[i], activation: 'relu'}));
    }

    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    const myOptimizer = tf.train.adam(.03) 
    model.compile({ loss: 'binaryCrossentropy', optimizer: myOptimizer });
    return model;
}

export { CustomCallback, trainModel, createModel, nodesValue, hiddenLayersValue};
