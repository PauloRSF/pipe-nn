use std::error::Error;

use pipe_nn::{activation::ActivationFunction, forward_values, LayerInput};

struct HiddenNeuron {
    pub bias: f64,
    pub input_weights: Vec<f64>,
    pub activation: ActivationFunction,
}

impl HiddenNeuron {
    pub fn fire(&self, values: &[f64]) -> f64 {
        assert_eq!(self.input_weights.len(), values.len());

        let weighted_sum: f64 = (0..values.len())
            .map(|i| self.input_weights[i] * values[i])
            .sum();

        self.activation.activate(weighted_sum + self.bias)
    }
}

fn compute_layer(neurons: &[HiddenNeuron], input_values: &[f64]) -> Vec<f64> {
    neurons
        .iter()
        .map(|neuron| neuron.fire(input_values))
        .collect::<Vec<_>>()
}

fn main() -> Result<(), Box<dyn Error>> {
    let layer_input = LayerInput::default();
    let layer_activation = ActivationFunction::Sigmoid;

    for input_values in layer_input {
        let neurons = vec![
            HiddenNeuron {
                bias: 10.0676,
                input_weights: vec![-6.6619, -6.3597],
                activation: layer_activation,
            },
            HiddenNeuron {
                bias: 2.8261,
                input_weights: vec![-5.9874, -9.9025],
                activation: layer_activation,
            },
        ];

        let output_values = compute_layer(&neurons, &input_values);

        forward_values(&output_values)?;
    }

    Ok(())
}
