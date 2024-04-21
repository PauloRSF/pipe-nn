use std::error::Error;

use pipe_nn::{activation::ActivationFunction, layer::Layer, layer::Neuron};

fn main() -> Result<(), Box<dyn Error>> {
    let activation = ActivationFunction::Sigmoid;

    let neurons = vec![
        Neuron {
            bias: 10.0676,
            input_weights: vec![-6.6619, -6.3597],
        },
        Neuron {
            bias: 2.8261,
            input_weights: vec![-5.9874, -9.9025],
        },
    ];

    Layer::new(neurons, activation).process()
}
