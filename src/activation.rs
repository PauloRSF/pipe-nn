use std::f64::consts;

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
}

impl ActivationFunction {
    pub fn activate(&self, value: f64) -> f64 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + consts::E.powf(-value)),
        }
    }
}
