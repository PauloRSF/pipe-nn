use std::error::Error;

use pipe_nn::{forward, layer::LayerInput};

fn main() -> Result<(), Box<dyn Error>> {
    let layer_input = LayerInput::default();

    for input_values in layer_input {
        forward(&input_values)?;
    }

    Ok(())
}
