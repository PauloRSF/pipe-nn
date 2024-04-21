use std::{error::Error, fs::read_to_string};

use pipe_nn::layer::LayerInput;

pub fn get_training_answers() -> Result<Vec<f64>, Box<dyn Error>> {
    Ok(read_to_string("training.txt")?
        .trim()
        .split('\n')
        .map(|vstr| vstr.parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let layer_input = LayerInput::default();
    let answers = get_training_answers()?;
    let input_values_with_revolving_index = layer_input
        .enumerate()
        .map(|(index, value)| (index % answers.len(), value))
        .collect::<Vec<_>>();

    for (index, input_values) in input_values_with_revolving_index {
        let error = (answers[index] - input_values[0]).powi(2);

        eprintln!("{:?} {:?} {:?}", answers[index], input_values[0], error);
    }

    Ok(())
}
