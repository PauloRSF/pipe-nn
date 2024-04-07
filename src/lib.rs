use std::{
    error::Error,
    io::{stdin, stdout, Write},
};

pub mod activation;

#[derive(Default)]
pub struct LayerInput {}

impl Iterator for LayerInput {
    type Item = Vec<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        read_layer_input().unwrap_or(None)
    }
}

pub fn read_layer_input() -> Result<Option<Vec<f64>>, Box<dyn Error>> {
    let mut serialized_values = String::new();

    if let Ok(0) = stdin().read_line(&mut serialized_values) {
        Ok(None)
    } else {
        let values = serialized_values
            .strip_suffix('\n')
            .unwrap_or(serialized_values.as_str())
            .split(' ')
            .map(|str| str.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Some(values))
    }
}

pub fn forward_values(output_values: &[f64]) -> std::io::Result<()> {
    let serialized_output_values = output_values
        .iter()
        .map(|weight| weight.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    stdout().write_all(format!("{}\n", serialized_output_values).as_bytes())?;

    Ok(())
}
