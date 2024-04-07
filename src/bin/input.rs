use std::error::Error;

use pipe_nn::{forward_values, LayerInput};

fn main() -> Result<(), Box<dyn Error>> {
    let layer_input = LayerInput::default();

    for input_values in layer_input {
        forward_values(&input_values)?;
    }

    Ok(())
}
