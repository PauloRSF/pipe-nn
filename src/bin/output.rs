use std::{error::Error, fs::read_to_string};

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

pub fn get_training_answers() -> Result<Vec<f64>, Box<dyn Error>> {
    Ok(read_to_string("training.txt")?
        .trim()
        .split("\n")
        .map(|vstr| vstr.parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let layer_input = LayerInput::default();
    let layer_activation = ActivationFunction::Sigmoid;
    let answers = get_training_answers()?;
    let input_values_with_revolving_index = layer_input
        .enumerate()
        .map(|(index, value)| (index % answers.len(), value))
        .collect::<Vec<_>>();

    for (index, input_values) in input_values_with_revolving_index {
        let neuron = OutputNeuron {
            bias: -4.6458,
            input_weights: vec![9.461, -9.9307],
            activation: layer_activation,
        };

        let output_value = neuron.fire(input_values);

        let error = answers[index] - output_value;

        eprintln!("{:?} {:?} {:?}", answers[index], output_value, error);

        forward_values(&[output_value])?;
    }

    Ok(())
}
