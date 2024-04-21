use std::{error::Error, io::stdin};

use crate::{activation::ActivationFunction, forward, parse_number_list};

pub struct Neuron {
    pub bias: f64,
    pub input_weights: Vec<f64>,
}

impl Neuron {
    pub fn fire(&self, activation: ActivationFunction, values: &[f64]) -> f64 {
        assert_eq!(self.input_weights.len(), values.len());

        let weighted_sum: f64 = (0..values.len())
            .map(|i| self.input_weights[i] * values[i])
            .sum();

        activation.activate(weighted_sum + self.bias)
    }
}

pub struct Layer {
    input: LayerInput,
    neurons: Vec<Neuron>,
    activation: ActivationFunction,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>, activation: ActivationFunction) -> Self {
        Self {
            neurons,
            activation,
            input: LayerInput::default(),
        }
    }

    pub fn process(&self) -> Result<(), Box<dyn Error>> {
        for input_values in self.input {
            let output_values = self
                .neurons
                .iter()
                .map(|neuron| neuron.fire(self.activation, input_values.as_slice()))
                .collect::<Vec<_>>();

            forward(&output_values)?;
        }

        Ok(())
    }
}

#[derive(Default, Clone, Copy)]
pub struct LayerInput {}

impl Iterator for LayerInput {
    type Item = Vec<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut serialized_values = String::new();

        match stdin().read_line(&mut serialized_values) {
            Ok(0) => None,
            Ok(_) => match parse_number_list(serialized_values) {
                Ok(values) => Some(values),
                _ => None,
            },
            _ => None,
        }
    }
}
