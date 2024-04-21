use std::{
    error::Error,
    fs::File,
    io::{self, stdout, Read, Write},
};

pub mod activation;
pub mod layer;

fn parse_number_list(serialized_values: String) -> Result<Vec<f64>, Box<dyn Error>> {
    Ok(serialized_values
        .strip_suffix('\n')
        .unwrap_or(serialized_values.as_str())
        .split(' ')
        .map(|str| str.parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()?)
}

fn serialize_number_list(values: &[f64]) -> String {
    values
        .iter()
        .map(f64::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn forward(output_values: &[f64]) -> std::io::Result<()> {
    let data = format!("{}\n", serialize_number_list(output_values));

    stdout().write_all(data.as_bytes())
}

fn open_pipe() -> io::Result<File> {
    File::open("pipe")
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
