use std::error::Error;

use pipe_nn::{activation::ActivationFunction, layer::Layer, layer::Neuron};

fn main() -> Result<(), Box<dyn Error>> {
    let activation = ActivationFunction::Sigmoid;

    let neurons = vec![Neuron {
        bias: -4.6458,
        input_weights: vec![9.461, -9.9307],
    }];

    Layer::new(neurons, activation).process()
}
