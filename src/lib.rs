use std::{
    error::Error,
    fs::File,
    io::{self, stdin, stdout, Read, Write},
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

fn open_pipe() -> io::Result<File> {
    File::open("/home/paulo/Code/projects/pipe-nn/pipe")
}

pub fn read_error() -> io::Result<f64> {
    let mut pipe = open_pipe()?;

    let mut data = String::new();

    loop {
        let bytes_read = pipe.read_to_string(&mut data)?;

        if bytes_read == 0 {
            continue;
        }

        return Ok(data.trim().parse::<f64>().unwrap());
    }
}

pub fn send_error(error: f64) -> io::Result<()> {
    let mut pipe = open_pipe()?;

    pipe.write_all(format!("{}\n", error).as_bytes())?;

    Ok(())
}
