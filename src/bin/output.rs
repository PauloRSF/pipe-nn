use std::error::Error;

use pipe_nn::{activation::ActivationFunction, forward_values, LayerInput};

struct OutputNeuron {
    pub bias: f64,
    pub input_weights: Vec<f64>,
    pub activation: ActivationFunction,
}

impl OutputNeuron {
    pub fn fire(&self, values: Vec<f64>) -> f64 {
        assert_eq!(self.input_weights.len(), values.len());

        let weighted_sum: f64 = (0..values.len())
            .map(|i| self.input_weights[i] * values[i])
            .sum();

        self.activation.activate(weighted_sum + self.bias)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let layer_input = LayerInput::default();
    let layer_activation = ActivationFunction::Sigmoid;

    for input_values in layer_input {
        let neuron = OutputNeuron {
            bias: -4.6458,
            input_weights: vec![9.461, -9.9307],
            activation: layer_activation,
        };

        let output_values = vec![neuron.fire(input_values)];

        forward_values(&output_values)?;
    }

    Ok(())
}
